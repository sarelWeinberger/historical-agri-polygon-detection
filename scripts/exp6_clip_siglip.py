#!/usr/bin/env python3
"""
Experiment 6 — CLIP / SigLIP contrastive tile scoring, zero-shot.

Slide overlapping tiles (224/336) over each clean crop, embed each tile, and
score it by the contrastive margin between the positive cultivation prompts and
the negative terrain prompts. Assemble a similarity heatmap, threshold it into
polygons, and evaluate vs the GT. Also reports a threshold-free tile-level AUROC
(cultivated tiles vs hard-negative tiles) — the cleanest "is there signal?" test.

CLIP and SigLIP are loaded one at a time with VRAM verified/freed between them.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, score_prediction, overlay,
                                    mask_to_polygons, Results, free_gpu, cuda_free_mb)
from src.evaluation import prompts as P

OUT = "outputs/zeroshot/exp6_clip_siglip"
DATA = "outputs/zeroshot/data"
MODELS = {
    "clip": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-base-patch16-224",
}
TILE_SIZES = [224, 336]


def tile_grid(H, W, size, overlap=0.5):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, W - size + 1), step)) or [0]
    ys = list(range(0, max(1, H - size + 1), step)) or [0]
    if xs[-1] != W - size and W > size: xs.append(W - size)
    if ys[-1] != H - size and H > size: ys.append(H - size)
    return [(x, y) for y in ys for x in xs]


def auroc(scores, labels):
    """labels: 1 positive / 0 negative. Rank-based AUROC."""
    s = np.asarray(scores); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(s) + 1)
    rpos = ranks[y == 1].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _as_tensor(f):
    """Coerce a features return (tensor or ModelOutput) to a 2D float tensor."""
    if torch.is_tensor(f):
        return f.float()
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        if hasattr(f, attr) and getattr(f, attr) is not None:
            return getattr(f, attr).float()
    return f.last_hidden_state.mean(1).float()


def encode_text(model, proc, kind, texts):
    inp = proc(text=texts, return_tensors="pt", padding="max_length"
               if kind == "siglip" else True).to("cuda")
    with torch.inference_mode():
        f = model.get_text_features(**inp)
    return torch.nn.functional.normalize(_as_tensor(f), dim=-1)


def run_model(kind, model_id, targets, res, dtype):
    from transformers import AutoProcessor, AutoModel
    with gpu_session(f"{kind}:{model_id.split('/')[-1]}") as sess:
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to("cuda").eval()
        pos_t = encode_text(model, proc, kind, P.CLIP_POSITIVE)
        neg_t = encode_text(model, proc, kind, P.CLIP_NEGATIVE)

        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            heat = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
            tile_scores, tile_labels = [], []
            t0 = time.time()
            for size in TILE_SIZES:
                coords = tile_grid(H, W, size)
                imgs = [rgb[y:y+size, x:x+size] for (x, y) in coords]
                # batch encode tiles
                feats = []
                for i in range(0, len(imgs), 64):
                    batch = imgs[i:i+64]
                    inp = proc(images=batch, return_tensors="pt").to("cuda")
                    inp["pixel_values"] = inp["pixel_values"].to(dtype)
                    with torch.inference_mode():
                        f = model.get_image_features(pixel_values=inp["pixel_values"])
                    feats.append(torch.nn.functional.normalize(_as_tensor(f), dim=-1))
                feats = torch.cat(feats, 0)
                pos_sim = (feats @ pos_t.T).mean(1)      # mean sim to positive prompts
                neg_sim = (feats @ neg_t.T).mean(1)
                margin = (pos_sim - neg_sim).cpu().numpy()
                for (x, y), sc in zip(coords, margin):
                    heat[y:y+size, x:x+size] += sc; wsum[y:y+size, x:x+size] += 1
                    cx, cy = x + size // 2, y + size // 2
                    lab = 1 if gt_pos[cy, cx] > 0 else (0 if gt_neg[cy, cx] > 0 else -1)
                    if lab >= 0:
                        tile_scores.append(float(sc)); tile_labels.append(lab)
            dt = time.time() - t0
            heat = heat / np.maximum(wsum, 1)
            hn = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            # threshold: Otsu on normalized heat
            thr_val = cv2.threshold((hn * 255).astype(np.uint8), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
            pred = (hn >= max(thr_val, 0.5)).astype(np.uint8)
            m = score_prediction(pred, gt_pos, gt_neg)
            tile_auc = auroc(tile_scores, tile_labels)
            res.add(model=kind, exp="exp6", model_id=model_id, target=t["name"], year=t["year"],
                    tile_auc=(round(tile_auc, 4) if tile_auc is not None else None),
                    n_tiles=len(tile_labels), runtime_s=round(dt, 3),
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
            # save heatmap + overlay
            hm = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm = cv2.addWeighted(clean, 0.5, hm, 0.5, 0)
            cv2.imwrite(os.path.join(OUT, "overlays", f"{t['name']}_{kind}_heat.png"), hm)
            overlay(clean, pred, gt_pos, gt_neg,
                    os.path.join(OUT, "overlays", f"{t['name']}_{kind}_pred.png"),
                    title=f"{kind} tiles cov={m['coverage']} iou={m['mask_iou']} auc={tile_auc}")
        free_gpu(model)


def main():
    os.makedirs(os.path.join(OUT, "overlays"), exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for kind, mid in MODELS.items():
        run_model(kind, mid, targets, res, dtype)
        free, total = cuda_free_mb()
        print(f"[transition] after {kind}: free VRAM {free:.0f}/{total:.0f} MB")

    rows = res.rows
    print(f"\n=== Exp6 CLIP/SigLIP tile scoring: {len(rows)} rows ===")
    for kind in MODELS:
        sub = [r for r in rows if r["model"] == kind]
        aucs = [r["tile_auc"] for r in sub if r["tile_auc"] is not None]
        print(f"  {kind:7s} mean tile-AUROC={np.mean(aucs):.3f}  "
              f"mean heat-IoU={np.mean([r['mask_iou'] for r in sub]):.3f}  "
              f"mean coverage={np.mean([r['coverage'] for r in sub if r['coverage'] is not None]):.3f}")


if __name__ == "__main__":
    main()
