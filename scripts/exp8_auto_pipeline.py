#!/usr/bin/env python3
"""
Experiment 8 — fully-automatic zero-shot pipeline:

    CLIP / DINOv2  ->  score map  ->  automatic region proposal  ->
    automatic SAM 2 prompts (point / points / box)  ->  SAM 2 mask  ->  polygon

NO ground truth is ever used to build a prompt. GT is used only to score.

Stage 1 (one model at a time): compute + cache a CLIP text-margin heatmap and a
label-free DINOv2 embedding map per target. Stage 2: load SAM 2 once and, for
every (target, source-map, proposal-method), propose regions, derive prompts,
segment, and score against the reference polygons.

DINOv2 is class-agnostic (no text), so its map is oriented label-free: PCA of the
tile embeddings gives the dominant 1-D texture axis, and its sign is fixed so the
*darker* tiles score high (cultivated fields are the dark, plough-textured
minority in these scans). This is an image-only heuristic — documented as such.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, score_prediction, boundary_f1,
                                    n_fragments, mask_to_polygons, free_gpu, cuda_free_mb)
from src.evaluation import prompts as P
from src.evaluation import region_proposal as RP

DATA = "outputs/zeroshot/data"
OUT = "outputs/zeroshot/exp8_auto"
MAPS = os.path.join(OUT, "maps")
METHODS = ["A_threshold", "B_localmax", "C_adaptive", "D_multithresh"]
PRIMARY_METHOD = "D_multithresh"
TOPK = 4  # keep at most K proposed regions (ranked by heat confidence)


def _tile_grid(H, W, size, overlap=0.5):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, W - size + 1), step)) or [0]
    ys = list(range(0, max(1, H - size + 1), step)) or [0]
    if W > size and xs[-1] != W - size: xs.append(W - size)
    if H > size and ys[-1] != H - size: ys.append(H - size)
    return [(x, y) for y in ys for x in xs]


def _as_tensor(f):
    if torch.is_tensor(f):
        return f.float()
    for a in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(f, a, None) is not None:
            return getattr(f, a).float()
    return f.last_hidden_state.mean(1).float()


# --------------------------------------------------------------------------- #
# Stage 1a — CLIP text-margin heatmap
def clip_maps(targets, dtype):
    from transformers import AutoProcessor, AutoModel
    with gpu_session("clip-vit-large") as s:
        proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = AutoModel.from_pretrained("openai/clip-vit-large-patch14",
                                          torch_dtype=dtype).to("cuda").eval()

        def emb_text(texts):
            inp = proc(text=texts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                f = model.get_text_features(**inp)
            return torch.nn.functional.normalize(_as_tensor(f), dim=-1)

        pos_t, neg_t = emb_text(P.CLIP_POSITIVE), emb_text(P.CLIP_NEGATIVE)
        for t in targets:
            rgb = cv2.cvtColor(cv2.imread(t["clean_path"]), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            heat = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
            for size in [224, 336]:
                coords = _tile_grid(H, W, size)
                imgs = [rgb[y:y+size, x:x+size] for (x, y) in coords]
                feats = []
                for i in range(0, len(imgs), 64):
                    inp = proc(images=imgs[i:i+64], return_tensors="pt").to("cuda")
                    inp["pixel_values"] = inp["pixel_values"].to(dtype)
                    with torch.inference_mode():
                        f = model.get_image_features(pixel_values=inp["pixel_values"])
                    feats.append(torch.nn.functional.normalize(_as_tensor(f), dim=-1))
                feats = torch.cat(feats, 0)
                margin = ((feats @ pos_t.T).mean(1) - (feats @ neg_t.T).mean(1)).cpu().numpy()
                for (x, y), sc in zip(coords, margin):
                    heat[y:y+size, x:x+size] += sc; wsum[y:y+size, x:x+size] += 1
            heat /= np.maximum(wsum, 1)
            np.save(os.path.join(MAPS, f"{t['name']}_clip.npy"), heat)
        free_gpu(model)


# Stage 1b — DINOv2 label-free embedding map
def dinov2_maps(targets, dtype):
    from transformers import AutoImageProcessor, AutoModel
    from sklearn.decomposition import PCA
    TILE, STR = 96, 48
    with gpu_session("dinov2-base") as s:
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base", dtype=dtype).to("cuda").eval()
        for t in targets:
            gray = cv2.imread(t["clean_path"], 0)
            rgb = cv2.cvtColor(cv2.imread(t["clean_path"]), cv2.COLOR_BGR2RGB)
            H, W = gray.shape
            coords = _tile_grid(H, W, TILE, overlap=0.5)
            embs, bright = [], []
            for i in range(0, len(coords), 128):
                batch = [cv2.resize(rgb[y:y+TILE, x:x+TILE], (224, 224)) for (x, y) in coords[i:i+128]]
                inp = proc(images=batch, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                with torch.inference_mode():
                    o = model(**inp)
                embs.append(torch.nn.functional.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy())
            for (x, y) in coords:
                bright.append(float(gray[y:y+TILE, x:x+TILE].mean()))
            E = np.concatenate(embs, 0); bright = np.array(bright)
            # dominant embedding axis (unsupervised)
            pc1 = PCA(n_components=1).fit_transform(E - E.mean(0)).ravel()
            # orient label-free: cultivated fields are the DARK minority -> score
            # should ANTI-correlate with brightness
            if np.corrcoef(pc1, bright)[0, 1] > 0:
                pc1 = -pc1
            score = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-6)
            heat = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
            for (x, y), sc in zip(coords, score):
                heat[y:y+TILE, x:x+TILE] += sc; wsum[y:y+TILE, x:x+TILE] += 1
            heat /= np.maximum(wsum, 1)
            np.save(os.path.join(MAPS, f"{t['name']}_dinov2.npy"), heat)
        free_gpu(model)


# Stage 2 — SAM 2 with automatic prompts
def sam2_stage(targets, results):
    from src.evaluation.sam2_helper import SAM2
    with gpu_session("sam2.1-hiera-large") as s:
        sam = SAM2()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            gt_area = max(int((gt_pos > 0).sum()), 1)
            min_area = 0.004 * H * W
            for source in ["clip", "dinov2"]:
                heat = np.load(os.path.join(MAPS, f"{t['name']}_{source}.npy"))
                hn = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
                for method in METHODS:
                    t0 = time.time()
                    regions = RP.propose(hn, method, min_area)[:TOPK]
                    pred = np.zeros((H, W), np.uint8)
                    n_prompts = 0
                    for region, conf in regions:
                        pr = RP.prompts_from_region(region, hn)
                        n_prompts += 1
                        # Prefer the box prompt (best in exp3). Fall back to multi-point
                        # only if SAM2's box mask collapses (tiny) or scores very low.
                        mb, sb = sam.segment_boxes(rgb, [pr["box"]])
                        sb = np.mean(sb) if sb else 0.0
                        bx = pr["box"]; box_area = max((bx[2]-bx[0])*(bx[3]-bx[1]), 1)
                        if (mb > 0).sum() < 0.15 * box_area or sb < 0.5:
                            mp, sp = sam.segment_points(rgb, pr["points"], [1] * len(pr["points"]))
                            best = mp if sp > sb else mb
                        else:
                            best = mb
                        pred |= (best > 0).astype(np.uint8)
                    # label-free speckle removal: drop connected components < min_area
                    nc, lab, st, _ = cv2.connectedComponentsWithStats(pred, 8)
                    clean_pred = np.zeros_like(pred)
                    for i in range(1, nc):
                        if st[i, cv2.CC_STAT_AREA] >= min_area:
                            clean_pred[lab == i] = 1
                    pred = clean_pred
                    dt = time.time() - t0
                    m = score_prediction(pred, gt_pos, gt_neg)
                    bf1 = boundary_f1(pred, gt_pos)
                    frag = n_fragments(pred)
                    polys = mask_to_polygons(pred, min_area=150)
                    row = dict(exp="exp8", pipeline=f"{source}->rp->sam2", source=source,
                               method=method, target=t["name"], year=t["year"],
                               n_regions=len(regions), n_polys=len(polys),
                               poly_iou=round(m["mask_iou"], 4), coverage=m["coverage"],
                               fp_on_black=m["fp_on_black"], fp_area_frac=round(m["fp_area_frac"], 4),
                               area_ratio=round((pred > 0).sum() / gt_area, 3),
                               fragments=frag, boundary_f1=round(bf1, 4), runtime_s=round(dt, 3))
                    results.append(row)
                    np.save(os.path.join(MAPS, f"{t['name']}_{source}_{method}_pred.npy"), pred)
                    json.dump({"polygons": polys}, open(
                        os.path.join(OUT, f"{t['name']}_{source}_{method}_polys.json"), "w"))
        free_gpu(sam.model)


def main():
    os.makedirs(MAPS, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    results = []

    have_clip = all(os.path.exists(os.path.join(MAPS, f"{t['name']}_clip.npy")) for t in targets)
    have_dino = all(os.path.exists(os.path.join(MAPS, f"{t['name']}_dinov2.npy")) for t in targets)
    if not have_clip:
        clip_maps(targets, dtype)
        print(f"[transition] free {cuda_free_mb()[0]:.0f} MB after CLIP")
    else:
        print("[cache] CLIP maps present, skipping CLIP load")
    if not have_dino:
        dinov2_maps(targets, dtype)
        print(f"[transition] free {cuda_free_mb()[0]:.0f} MB after DINOv2")
    else:
        print("[cache] DINOv2 maps present, skipping DINOv2 load")
    sam2_stage(targets, results)

    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\n=== Exp8 auto pipeline ({len(results)} rows) ===")
    for source in ["clip", "dinov2"]:
        print(f"\n{source.upper()} -> region-proposal -> SAM2:")
        for method in METHODS:
            sub = [r for r in results if r["source"] == source and r["method"] == method]
            iou = np.mean([r["poly_iou"] for r in sub])
            cov = np.mean([r["coverage"] for r in sub if r["coverage"] is not None])
            bf = np.mean([r["boundary_f1"] for r in sub])
            ar = np.mean([r["area_ratio"] for r in sub])
            fr = np.mean([r["fragments"] for r in sub])
            print(f"  {method:15s} IoU={iou:.3f} cov={cov:.3f} bF1={bf:.3f} "
                  f"areaRatio={ar:.2f} frags={fr:.1f}")


if __name__ == "__main__":
    main()
