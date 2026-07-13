#!/usr/bin/env python3
"""
Experiment 3 — Prompted SAM 2 with manual (human-style) prompts.

For each cultivated GT field we simulate a human operator and compare four prompt
styles, scoring the returned mask against that field:
    A) single positive point (field centroid)
    B) multiple positive points (sampled inside the field)
    C) positive + negative points (negatives on hard-negative / background)
    D) bounding box around the field

This answers: *given a human prompt, can SAM 2 recover accurate field boundaries?*
The GT geometry is used only to place prompts and to score — never to train.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, _iou, overlay, mask_to_polygons,
                                    Results, free_gpu)
from src.evaluation.sam2_helper import SAM2

OUT = "outputs/zeroshot/exp3_prompted_sam2"
DATA = "outputs/zeroshot/data"


def poly_mask(points, holes, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(points, np.int32)], 1)
    for hl in holes or []:
        cv2.fillPoly(m, [np.array(hl, np.int32)], 0)
    return m


def sample_inside(mask, k):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    idx = np.linspace(0, len(xs) - 1, k).astype(int)
    return [[int(xs[i]), int(ys[i])] for i in idx]


def centroid(mask):
    ys, xs = np.where(mask > 0)
    return [int(xs.mean()), int(ys.mean())]


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))

    with gpu_session("sam2.1-hiera-large") as sess:
        sam = SAM2()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            pos_polys = [p for p in t["polygons"] if p["class"] == "cultivated_area"]
            neg_polys = [p for p in t["polygons"] if p["class"] == "hard_negative"]
            neg_pts = [centroid(poly_mask(p["points"], p.get("holes"), H, W)) for p in neg_polys]

            for fi, p in enumerate(pos_polys):
                fm = poly_mask(p["points"], p.get("holes"), H, W)
                if fm.sum() < 200:
                    continue
                cen = centroid(fm)
                multi = sample_inside(fm, 5)
                x, y, ww, hh = cv2.boundingRect(np.array(p["points"], np.int32))
                gbox = [x, y, x + ww, y + hh]

                def run_variant(name):
                    if name == "A_single_point":
                        return sam.segment_points(rgb, [cen], [1])
                    if name == "B_multi_points":
                        return sam.segment_points(rgb, multi, [1] * len(multi))
                    if name == "C_pos_neg_points":
                        npts = neg_pts or []
                        return sam.segment_points(rgb, multi + npts,
                                                  [1] * len(multi) + [0] * len(npts))
                    # D_box
                    mask, sc = sam.segment_boxes(rgb, [gbox])
                    return mask, (float(np.mean(sc)) if sc else None)

                for name in ["A_single_point", "B_multi_points", "C_pos_neg_points", "D_box"]:
                    t0 = time.time()
                    mask, score = run_variant(name)
                    dt = time.time() - t0
                    iou = _iou(mask, fm)
                    # false-positive leakage onto hard negatives
                    fpb = float(np.logical_and(mask > 0, gt_neg > 0).sum() / max((gt_neg > 0).sum(), 1))
                    res.add(model="sam2-prompted", exp="exp3", target=t["name"], year=t["year"],
                            field=fi, variant=name, field_iou=round(iou, 4),
                            sam_score=round(score, 4) if score else None,
                            fp_on_black=round(fpb, 4), runtime_s=round(dt, 3),
                            field_area_px=int(fm.sum()))
                    if name == "D_box":
                        overlay(clean, mask, fm, gt_neg,
                                os.path.join(OUT, "overlays", f"{t['name']}_field{fi}_{name}.png"),
                                pred_boxes=[[x, y, x+ww, y+hh]],
                                title=f"SAM2 {name} field{fi} IoU={iou:.3f}")
        sam_id = sam.model_id
        free_gpu(sam.model)

    rows = res.rows
    print(f"\n=== Exp3 Prompted SAM2 ({sam_id}): {len(rows)} rows ===")
    for v in ["A_single_point", "B_multi_points", "C_pos_neg_points", "D_box"]:
        sub = [r for r in rows if r["variant"] == v]
        if sub:
            miou = np.mean([r["field_iou"] for r in sub])
            mfpb = np.mean([r["fp_on_black"] for r in sub])
            print(f"  {v:18s} mean field IoU={miou:.3f}  mean fp_on_black={mfpb:.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
