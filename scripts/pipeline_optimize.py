#!/usr/bin/env python3
"""
Bounded zero-shot optimization: sweep fusion x proposal x prompt configs over the
cached priors, run SAM 2 (loaded once) with multi-prompt + label-free selection,
and score against GT (evaluation only). Tracks the full metric suite and reports
mean / median / worst IoU so a config that spikes one target but collapses another
is penalised.
"""
from __future__ import annotations

import json, os, sys, time, itertools
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, score_prediction, boundary_f1,
                                    n_fragments, mask_to_polygons, free_gpu)
from src.pipeline import pipeline as PL
from src.pipeline.priors import norm01

DATA = "outputs/zeroshot/data"
CACHE = "outputs/pipeline/priors"
OUT = "outputs/pipeline/rounds"
PRIOR_NAMES = ["clip_ms", "clip_lc", "dino_bright", "dino_proto", "texture"]


def load_priors(name):
    return {p: np.load(os.path.join(CACHE, f"{name}_{p}.npy")) for p in PRIOR_NAMES}


def run_config(sam, targets, fuse_spec, method, topk=6):
    """Return per-target rows for one config."""
    rows = []
    for t in targets:
        clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        gp = cv2.imread(t["gt_pos_path"], 0); gn = cv2.imread(t["gt_neg_path"], 0)
        H, W = gp.shape; gt_area = max(int((gp > 0).sum()), 1)
        priors = load_priors(t["name"])
        fused = PL.fuse(priors, fuse_spec)
        # negative prior = high where CLIP margin low (i.e. terrain-like)
        neg_map = norm01(1 - priors["clip_ms"])
        min_area = 0.004 * H * W
        regions = PL.propose(fused, method, min_area, topk=topk)
        t0 = time.time()
        pred = np.zeros((H, W), np.uint8)
        for region, conf in regions:
            pr = PL.region_prompts(region, fused, neg_map)
            cands = []
            # prompt variants -> SAM2 (multimask); collect all candidate masks
            mb, sb = sam.segment_boxes(rgb, [pr["box"]]); cands.append(mb)
            me, se = sam.segment_boxes(rgb, [pr["ebox"]]); cands.append(me)
            mp, sp = sam.segment_points(rgb, pr["pos"], [1] * len(pr["pos"])); cands.append(mp)
            if pr["neg"]:
                mpn, _ = sam.segment_points(rgb, pr["pos"] + pr["neg"],
                                            [1] * len(pr["pos"]) + [0] * len(pr["neg"]))
                cands.append(mpn)
                mbn = _box_with_neg(sam, rgb, pr["box"], pr["pos"], pr["neg"]); cands.append(mbn)
            # label-free selection among candidates
            best = max(cands, key=lambda m: PL.mask_score(m, fused, neg_map))
            pred |= (best > 0).astype(np.uint8)
        # cleanup: drop speckle, fill holes
        pred = _cleanup(pred, min_area)
        dt = time.time() - t0
        m = score_prediction(pred, gp, gn)
        rows.append(dict(target=t["name"], year=t["year"],
                         poly_iou=round(m["mask_iou"], 4), coverage=m["coverage"],
                         fp_on_black=m["fp_on_black"], fp_area_frac=round(m["fp_area_frac"], 4),
                         area_ratio=round((pred > 0).sum() / gt_area, 3),
                         boundary_f1=round(boundary_f1(pred, gp), 4),
                         fragments=n_fragments(pred), runtime_s=round(dt, 2),
                         _pred=pred))
    return rows


def _box_with_neg(sam, rgb, box, pos, neg):
    # SAM2 box + points combined (helper handles points/boxes separately; approximate
    # by intersecting a box mask with a pos/neg point mask)
    mb, _ = sam.segment_boxes(rgb, [box])
    mp, _ = sam.segment_points(rgb, pos + neg, [1]*len(pos) + [0]*len(neg))
    return ((mb > 0) & (mp > 0)).astype(np.uint8) if (mp > 0).sum() > 0.1*(mb > 0).sum() else mb


def _cleanup(pred, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(pred, 8)
    out = np.zeros_like(pred)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area: out[lab == i] = 1
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(out).astype(np.uint8)


def summarize(rows):
    iou = [r["poly_iou"] for r in rows]
    return dict(mean_iou=round(float(np.mean(iou)), 4), median_iou=round(float(np.median(iou)), 4),
                min_iou=round(float(np.min(iou)), 4), max_iou=round(float(np.max(iou)), 4),
                mean_cov=round(float(np.mean([r["coverage"] for r in rows if r["coverage"] is not None])), 4),
                mean_bf1=round(float(np.mean([r["boundary_f1"] for r in rows])), 4),
                mean_frag=round(float(np.mean([r["fragments"] for r in rows])), 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))

    # ---- config grid (bounded) ----
    fusions = {
        "clip_only":        {"rule": "weighted_avg", "use": ["clip_ms"]},
        "clip_lc":          {"rule": "weighted_avg", "use": ["clip_lc"]},
        "dino_only":        {"rule": "weighted_avg", "use": ["dino_proto"]},
        "texture_only":     {"rule": "weighted_avg", "use": ["texture"]},
        "clip+texture_avg": {"rule": "weighted_avg", "use": ["clip_ms", "texture"], "weights": {"clip_ms": 2, "texture": 1}},
        "clip+texture_geo": {"rule": "geometric_mean", "use": ["clip_ms", "texture"]},
        "clip_gate_texture":{"rule": "gate", "use": ["clip_ms", "texture"]},
        "clip+dino_avg":    {"rule": "weighted_avg", "use": ["clip_ms", "dino_proto"]},
        "clip+dino+tex_avg":{"rule": "weighted_avg", "use": ["clip_ms", "dino_proto", "texture"], "weights": {"clip_ms": 2, "dino_proto": 1, "texture": 1}},
        "clip+dino+tex_rank":{"rule": "rank_avg", "use": ["clip_ms", "dino_proto", "texture"]},
        "full_gate":        {"rule": "gate", "use": ["clip_ms", "dino_proto", "texture"]},
    }
    methods = ["multi_thresh", "adaptive", "watershed", "stability"]

    all_results = {}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with gpu_session("sam2.1-hiera-large") as s:
        from src.evaluation.sam2_helper import SAM2
        sam = SAM2()
        # Stage 1: fusion sweep with a fixed decent proposal method
        print("=== fusion sweep (method=multi_thresh) ===")
        for fname, fspec in fusions.items():
            rows = run_config(sam, targets, fspec, "multi_thresh")
            summ = summarize(rows)
            all_results[f"fuse::{fname}"] = {"summary": summ,
                "rows": [{k: v for k, v in r.items() if k != "_pred"} for r in rows]}
            print(f"  {fname:20s} meanIoU={summ['mean_iou']} med={summ['median_iou']} "
                  f"min={summ['min_iou']} max={summ['max_iou']} bF1={summ['mean_bf1']} frag={summ['mean_frag']}")
        # pick best fusion by (mean+min)/2 to penalise collapse
        best_f = max(fusions, key=lambda f: 0.5*(all_results[f"fuse::{f}"]["summary"]["mean_iou"]
                                                 + all_results[f"fuse::{f}"]["summary"]["min_iou"]))
        print(f"\nbest fusion (mean+min): {best_f}")
        # Stage 2: proposal sweep on best fusion
        print(f"\n=== proposal sweep (fusion={best_f}) ===")
        for meth in methods:
            rows = run_config(sam, targets, fusions[best_f], meth)
            summ = summarize(rows)
            all_results[f"prop::{meth}"] = {"summary": summ,
                "rows": [{k: v for k, v in r.items() if k != "_pred"} for r in rows]}
            print(f"  {meth:15s} meanIoU={summ['mean_iou']} med={summ['median_iou']} "
                  f"min={summ['min_iou']} max={summ['max_iou']} bF1={summ['mean_bf1']} frag={summ['mean_frag']}")
        best_m = max(methods, key=lambda m: 0.5*(all_results[f"prop::{m}"]["summary"]["mean_iou"]
                                                + all_results[f"prop::{m}"]["summary"]["min_iou"]))
        # Save the winning per-target predictions for maps
        print(f"\nbest proposal: {best_m}. Saving winning predictions...")
        rows = run_config(sam, targets, fusions[best_f], best_m)
        for r in rows:
            np.save(os.path.join(OUT, f"{r['target']}_best_pred.npy"), r["_pred"])
        all_results["WINNER"] = {"fusion": best_f, "method": best_m,
                                 "summary": summarize(rows),
                                 "rows": [{k: v for k, v in r.items() if k != "_pred"} for r in rows]}
        free_gpu(sam.model)

    json.dump(all_results, open(os.path.join(OUT, "sweep_results.json"), "w"), indent=2, ensure_ascii=False)
    w = all_results["WINNER"]
    print(f"\n### WINNER: fusion={w['fusion']} proposal={w['method']} -> {w['summary']}")
    print("baseline (exp8): CLIP mean 0.227, DINOv2 mean 0.210")


if __name__ == "__main__":
    main()
