#!/usr/bin/env python3
"""
Experiment 2 — Grounding DINO boxes -> SAM 2 boundary refinement.

Demonstrates the strict single-GPU discipline: load Grounding DINO, collect its
boxes for all targets, FREE it and verify VRAM, THEN load SAM 2 and segment.
SAM 2 only refines boundaries; it never decides cultivation.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, score_prediction, gt_boxes_from_mask,
                                    overlay, mask_to_polygons, Results, free_gpu, cuda_free_mb)
from src.evaluation.sam2_helper import SAM2

OUT = "outputs/zeroshot/exp2_gdino_sam2"
DATA = "outputs/zeroshot/data"
GDINO_ID = "IDEA-Research/grounding-dino-base"
PROMPTS = ["cultivated agricultural field", "field with cultivation rows"]
THRESHOLDS = [0.25, 0.35]


def collect_gdino_boxes(targets):
    """Load GDINO, return {target: {(prompt,thr): [boxes]}}, then free it."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    out = {}
    with gpu_session("grounding-dino-base(box-source)") as sess:
        proc = AutoProcessor.from_pretrained(GDINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to("cuda").eval()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            out[t["name"]] = {}
            for prompt in PROMPTS:
                inp = proc(images=pil, text=prompt + ".", return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    o = model(**inp)
                post = proc.post_process_grounded_object_detection(
                    o, inp["input_ids"], threshold=0.10, text_threshold=0.10,
                    target_sizes=[(H, W)])[0]
                boxes = post["boxes"].cpu().numpy().tolist()
                scores = post["scores"].cpu().numpy().tolist()
                for thr in THRESHOLDS:
                    out[t["name"]][(prompt, thr)] = [b for b, s in zip(boxes, scores) if s >= thr]
        free_gpu(model)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))

    boxes_by_target = collect_gdino_boxes(targets)
    free, total = cuda_free_mb()
    print(f"[transition] GDINO freed; free VRAM {free:.0f}/{total:.0f} MB before SAM2")

    with gpu_session("sam2.1-hiera-large") as sess:
        sam = SAM2()
        for t in targets:
            clean = cv2.imread(t["clean_path"])
            rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape; gt_boxes = gt_boxes_from_mask(gt_pos)
            for prompt in PROMPTS:
                for thr in THRESHOLDS:
                    boxes = boxes_by_target[t["name"]][(prompt, thr)]
                    t0 = time.time()
                    mask, scores = sam.segment_boxes(rgb, boxes)
                    dt = time.time() - t0
                    m = score_prediction(mask, gt_pos, gt_neg, pred_boxes=boxes, gt_boxes=gt_boxes)
                    res.add(model="gdino+sam2", exp="exp2", target=t["name"], year=t["year"],
                            prompt=prompt, threshold=thr, n_boxes=len(boxes),
                            sam_score=round(float(np.mean(scores)), 4) if scores else None,
                            runtime_s=round(dt, 3),
                            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
                    if thr == 0.25 and prompt == PROMPTS[0]:
                        polys = mask_to_polygons(mask)
                        overlay(clean, mask, gt_pos, gt_neg,
                                os.path.join(OUT, "overlays", f"{t['name']}__{prompt.replace(' ','_')}.png"),
                                pred_boxes=boxes,
                                title=f"GDINO+SAM2 '{prompt}' thr{thr} cov={m['coverage']} iou={m['mask_iou']} fpblk={m['fp_on_black']}")
                        json.dump({"type": "FeatureCollection", "features": [
                            {"type": "Feature", "properties": {"model": "gdino+sam2", "prompt": prompt},
                             "geometry": {"type": "Polygon", "coordinates": [[*[list(map(int, pt)) for pt in poly], list(map(int, poly[0]))]]}}
                            for poly in polys]}, open(os.path.join(OUT, f"{t['name']}_pred.geojson"), "w"))
        sam_id = sam.model_id
        free_gpu(sam.model)

    print(f"\n=== Exp2 GDINO+SAM2 ({sam_id}): {len(res)} rows ===")
    for t in targets:
        sub = [r for r in res.rows if r["target"] == t["name"] and r["threshold"] == 0.25
               and r["prompt"] == PROMPTS[0]]
        if sub:
            r = sub[0]
            print(f"  {t['name']:14s} cov={r['coverage']} iou={r['mask_iou']} "
                  f"fp_blk={r['fp_on_black']} predArea={r['pred_area_frac']:.2f} nbox={r['n_boxes']}")


if __name__ == "__main__":
    main()
