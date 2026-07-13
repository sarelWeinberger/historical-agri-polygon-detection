#!/usr/bin/env python3
"""
Generate per-model *polygon* predictions for every evaluation target, so they can
be drawn on comparison maps. One model loaded at a time (VRAM verified/freed).

Each model's representative zero-shot configuration is converted to a binary mask
and then to simplified polygons. Output: outputs/zeroshot/polygons/geometry.json
plus per-target/-model GeoJSON (pixel coords; ITM/EPSG:2039 added for the two
georeferenceable map sheets).

Florence-2 geometry is produced separately (isolated venv) by the same script's
`--florence-only` mode; results are merged into the same geometry.json.
"""
from __future__ import annotations

import argparse, json, os, sys
import cv2, numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, boxes_to_mask, polys_to_mask,
                                    mask_to_polygons, score_prediction, gt_boxes_from_mask,
                                    free_gpu, cuda_free_mb, _iou)
from src.evaluation import prompts as P

DATA = "outputs/zeroshot/data"
OUTP = "outputs/zeroshot/polygons"
GEOM = os.path.join(OUTP, "geometry.json")


def load_targets():
    return json.load(open(os.path.join(DATA, "eval_targets.json")))


def load_geom():
    return json.load(open(GEOM)) if os.path.exists(GEOM) else {}


def save_geom(g):
    os.makedirs(OUTP, exist_ok=True)
    json.dump(g, open(GEOM, "w"), indent=2, ensure_ascii=False)


def add(geom, target, model, mask, gt_pos, gt_neg, extra=None):
    polys = mask_to_polygons(mask, min_area=150)
    m = score_prediction(mask, gt_pos, gt_neg)
    geom.setdefault(target, {})[model] = {
        "polygons": polys,
        "coverage": m["coverage"], "mask_iou": round(m["mask_iou"], 4),
        "fp_on_black": m["fp_on_black"], "pred_area_frac": round(m["pred_area_frac"], 4),
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
def gdino_and_sam2(targets, geom):
    """GDINO boxes -> (a) GDINO polygons, (b) SAM2-refined polygons, and
    prompted-SAM2 (GT box) polygons. GDINO freed before SAM2 loads."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    boxes_map = {}
    with gpu_session("grounding-dino-base") as s:
        proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base").to("cuda").eval()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            inp = proc(images=pil, text="cultivated agricultural field.", return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = model(**inp)
            post = proc.post_process_grounded_object_detection(
                o, inp["input_ids"], threshold=0.10, text_threshold=0.10, target_sizes=[(H, W)])[0]
            b = post["boxes"].cpu().numpy().tolist(); sc = post["scores"].cpu().numpy().tolist()
            boxes_map[t["name"]] = [bb for bb, ss in zip(b, sc) if ss >= 0.25]
        free_gpu(model)

    free, total = cuda_free_mb(); print(f"[transition] free {free:.0f} MB before SAM2")
    from src.evaluation.sam2_helper import SAM2
    with gpu_session("sam2.1-hiera-large") as s:
        sam = SAM2()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            gboxes = boxes_map[t["name"]]
            add(geom, t["name"], "grounding_dino", boxes_to_mask(gboxes, H, W), gt_pos, gt_neg)
            gmask, _ = sam.segment_boxes(rgb, gboxes)
            add(geom, t["name"], "gdino_sam2", gmask, gt_pos, gt_neg)
            # prompted SAM2: box prompt per GT field (human-prompt upper bound)
            pm = np.zeros((H, W), np.uint8)
            for gb in gt_boxes_from_mask(gt_pos):
                mk, _ = sam.segment_boxes(rgb, [gb]); pm |= mk
            add(geom, t["name"], "prompted_sam2_box", pm, gt_pos, gt_neg,
                extra={"note": "human box prompt (upper bound)"})
        free_gpu(sam.model)


def owlv2(targets, geom):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    dtype = torch.bfloat16
    with gpu_session("owlv2-base") as s:
        proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble", torch_dtype=dtype).to("cuda").eval()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            keep = []
            for prompt in P.POSITIVE:                       # union of all positive prompts
                inp = proc(text=[[prompt]], images=pil, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                with torch.inference_mode():
                    o = model(**inp)
                post = proc.post_process_grounded_object_detection(
                    o, threshold=0.05, target_sizes=torch.tensor([[H, W]]))[0]
                keep += post["boxes"].float().cpu().numpy().tolist()
            add(geom, t["name"], "owlv2", boxes_to_mask(keep, H, W), gt_pos, gt_neg)
        free_gpu(model)


def _tile_grid(H, W, size, overlap=0.5):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, W - size + 1), step)) or [0]
    ys = list(range(0, max(1, H - size + 1), step)) or [0]
    if W > size and xs[-1] != W - size: xs.append(W - size)
    if H > size and ys[-1] != H - size: ys.append(H - size)
    return [(x, y) for y in ys for x in xs]


def _heatmap_polys(heat, gt_pos, gt_neg, geom, target, model):
    hn = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    thr = cv2.threshold((hn * 255).astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    pred = (hn >= max(thr, 0.5)).astype(np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    add(geom, target, model, pred, gt_pos, gt_neg)


def _as_tensor(f):
    if torch.is_tensor(f):
        return f.float()
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(f, attr, None) is not None:
            return getattr(f, attr).float()
    return f.last_hidden_state.mean(1).float()


def clip_siglip(targets, geom):
    from transformers import AutoProcessor, AutoModel
    dtype = torch.bfloat16
    for kind, mid in [("clip", "openai/clip-vit-large-patch14"),
                      ("siglip", "google/siglip-base-patch16-224")]:
        with gpu_session(f"{kind}") as s:
            proc = AutoProcessor.from_pretrained(mid)
            model = AutoModel.from_pretrained(mid, torch_dtype=dtype).to("cuda").eval()

            def emb_text(texts):
                pad = "max_length" if kind == "siglip" else True
                inp = proc(text=texts, return_tensors="pt", padding=pad).to("cuda")
                with torch.inference_mode():
                    f = model.get_text_features(**inp)
                return torch.nn.functional.normalize(_as_tensor(f), dim=-1)

            pos_t, neg_t = emb_text(P.CLIP_POSITIVE), emb_text(P.CLIP_NEGATIVE)
            for t in targets:
                clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
                gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
                H, W = gt_pos.shape
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
                _heatmap_polys(heat, gt_pos, gt_neg, geom, t["name"], kind)
            free_gpu(model)


def dinov2(targets, geom):
    """Prototype-similarity heatmap -> polygons. Uses GT tiles to form the
    cultivation prototype, so it is LABEL-INFORMED (analysis / reference only)."""
    from transformers import AutoImageProcessor, AutoModel
    dtype = torch.bfloat16
    TILE, STR = 112, 56
    with gpu_session("dinov2-base") as s:
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base", dtype=dtype).to("cuda").eval()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            coords = _tile_grid(H, W, TILE, overlap=0.5)
            embs = []
            for i in range(0, len(coords), 128):
                batch = [cv2.resize(rgb[y:y+TILE, x:x+TILE], (224, 224)) for (x, y) in coords[i:i+128]]
                inp = proc(images=batch, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                with torch.inference_mode():
                    o = model(**inp)
                embs.append(torch.nn.functional.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
            E = torch.cat(embs, 0)
            labs = []
            for (x, y) in coords:
                fp = (gt_pos[y:y+TILE, x:x+TILE] > 0).mean(); fn = (gt_neg[y:y+TILE, x:x+TILE] > 0).mean()
                labs.append(1 if fp > 0.5 else (0 if fn > 0.5 else -1))
            labs = np.array(labs)
            if (labs == 1).sum() == 0:
                continue
            pm = E[labs == 1].mean(0, keepdim=True)
            pm = torch.nn.functional.normalize(pm, dim=-1)
            score = (E @ pm.T).squeeze(1).numpy()
            heat = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
            for (x, y), sc in zip(coords, score):
                heat[y:y+TILE, x:x+TILE] += sc; wsum[y:y+TILE, x:x+TILE] += 1
            heat /= np.maximum(wsum, 1)
            _heatmap_polys(heat, gt_pos, gt_neg, geom, t["name"], "dinov2")
            geom[t["name"]]["dinov2"]["note"] = "label-informed prototype (analysis only)"
        free_gpu(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--florence-only", action="store_true")
    a = ap.parse_args()
    targets = load_targets()
    geom = load_geom()
    if a.florence_only:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from florence_geometry import run_florence
        run_florence(targets, geom, add)
    else:
        gdino_and_sam2(targets, geom)
        owlv2(targets, geom)
        clip_siglip(targets, geom)
        dinov2(targets, geom)
    save_geom(geom)
    print(f"\ngeometry -> {GEOM}")
    for tname, models in geom.items():
        print(f"  {tname[:16]:16s}: " + ", ".join(
            f"{m}({len(d['polygons'])}p,iou={d['mask_iou']})" for m, d in models.items()))


if __name__ == "__main__":
    main()
