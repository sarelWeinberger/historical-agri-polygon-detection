#!/usr/bin/env python3
"""
Render the required Phase-9 maps for every target:
  8-panel comparison — GT only / Exp8 baseline / best CLIP-only / best DINOv2-only /
  CLIP+DINOv2 fusion / full fusion+texture / best automatic SAM2 polygon / human ceiling
and the full flow figure:
  Original -> CLIP -> DINOv2 -> texture -> fused -> proposals -> SAM prompts ->
  SAM mask -> selected polygon -> GT comparison.

Reads cached priors, the sweep's winning predictions, and geometry.json (for the
Exp8 baseline and human-prompted SAM2 polygons).
"""
from __future__ import annotations

import json, os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import pipeline as PL
from src.pipeline.priors import norm01

DATA = "outputs/zeroshot/data"
CACHE = "outputs/pipeline/priors"
ROUNDS = "outputs/pipeline/rounds"
GEOM = "outputs/zeroshot/polygons/geometry.json"
OUT = "outputs/pipeline/maps"


def heat_rgb(clean, h):
    hn = (norm01(h) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(clean, 0.5, cv2.applyColorMap(hn, cv2.COLORMAP_JET), 0.5, 0),
                        cv2.COLOR_BGR2RGB)


def gt_over(rgb, gp, gn):
    v = rgb.copy()
    for m, c in [(gp, (0, 255, 0)), (gn, (0, 255, 255))]:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cs, -1, c, 2)
    return v


def pred_over(clean, pred, gp, gn):
    v = clean.copy(); pr = pred > 0; pos = gp > 0; neg = gn > 0
    lay = v.copy()
    lay[np.logical_and(pr, pos)] = (0, 200, 0)
    lay[np.logical_and(pr, np.logical_not(np.logical_or(pos, neg)))] = (0, 0, 230)
    lay[np.logical_and(pr, neg)] = (230, 60, 0)
    v = cv2.addWeighted(lay, 0.45, v, 0.55, 0)
    return cv2.cvtColor(gt_over(cv2.cvtColor(v, cv2.COLOR_BGR2RGB), gp, gn), cv2.COLOR_RGB2BGR) \
        if False else _draw_gt_bgr(v, gp, gn)


def _draw_gt_bgr(v, gp, gn):
    for m, c in [(gp, (0, 255, 0)), (gn, (0, 255, 255))]:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cs, -1, c, 2)
    return cv2.cvtColor(v, cv2.COLOR_BGR2RGB)


def polys_over(rgb, polys, gp, gn, col=(255, 255, 255)):
    v = rgb.copy()
    for p in polys:
        pts = np.array(p, np.int32).reshape(-1, 1, 2)
        ov = v.copy(); cv2.fillPoly(ov, [pts], col); v = cv2.addWeighted(ov, .35, v, .65, 0)
        cv2.polylines(v, [pts], True, col, 3, cv2.LINE_AA)
    return gt_over(v, gp, gn)


def iou(a, b):
    a = a > 0; b = b > 0
    return float(np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-6))


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    geom = json.load(open(GEOM))
    sweep = json.load(open(os.path.join(ROUNDS, "sweep_results.json")))
    win = sweep["WINNER"]
    fuse_full = {"rule": "weighted_avg", "use": ["clip_ms", "dino_proto", "texture"],
                 "weights": {"clip_ms": 2, "dino_proto": 1, "texture": 1}}

    for t in targets:
        name = t["name"]
        clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        gp = cv2.imread(t["gt_pos_path"], 0); gn = cv2.imread(t["gt_neg_path"], 0)
        pri = {p: np.load(os.path.join(CACHE, f"{name}_{p}.npy"))
               for p in ["clip_ms", "clip_lc", "dino_proto", "texture"]}
        best_pred = np.load(os.path.join(ROUNDS, f"{name}_best_pred.npy"))
        gm = geom.get(name, {})
        full = PL.fuse(pri, fuse_full)

        def load_pred(v):
            p = os.path.join(ROUNDS, f"{name}_{v}_pred.npy")
            return np.load(p) if os.path.exists(p) else np.zeros_like(gp)

        def exp8_base():
            p = f"outputs/zeroshot/exp8_auto/maps/{name}_clip_A_threshold_pred.npy"
            return np.load(p) if os.path.exists(p) else np.zeros_like(gp)

        base8 = exp8_base(); clipp = load_pred("clip_only"); dinop = load_pred("dino_only")
        cdp = load_pred("clipdino"); ftp = load_pred("fulltex")
        best_iou = iou(best_pred, gp)

        panels = [
            ("Ground truth", gt_over(rgb, gp, gn)),
            (f"Exp8 baseline (IoU {iou(base8, gp):.3f})", pred_over(clean, base8, gp, gn)),
            (f"Best CLIP-only (IoU {iou(clipp, gp):.3f})", pred_over(clean, clipp, gp, gn)),
            (f"Best DINOv2-only (IoU {iou(dinop, gp):.3f})", pred_over(clean, dinop, gp, gn)),
            (f"CLIP+DINOv2 fusion (IoU {iou(cdp, gp):.3f})", pred_over(clean, cdp, gp, gn)),
            (f"Full fusion +texture (IoU {iou(ftp, gp):.3f})", pred_over(clean, ftp, gp, gn)),
            (f"WINNER: CLIP+texture auto (IoU {best_iou:.3f})", pred_over(clean, best_pred, gp, gn)),
            (f"Human-prompted SAM2 ceiling (IoU {gm.get('prompted_sam2_box',{}).get('mask_iou','-')})",
             polys_over(rgb, gm.get("prompted_sam2_box", {}).get("polygons", []), gp, gn)),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.ravel()
        for ax, (ttl, im) in zip(axes, panels):
            ax.imshow(im); ax.set_title(ttl, fontsize=10); ax.axis("off")
        fig.suptitle(f"{name} ({t['year']}) — fused zero-shot pipeline vs baseline & human ceiling  "
                     f"[winner: {win['fusion']} / {win['method']}]", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(OUT, f"{name}_compare8.png"), dpi=90); plt.close()

        # ---- full flow ----
        neg_map = norm01(1 - pri["clip_ms"])
        min_area = 0.004 * gp.shape[0] * gp.shape[1]
        regions = PL.propose(full, win["method"], min_area, topk=6)
        regvis = rgb.copy()
        for reg, _c in regions:
            cs, _ = cv2.findContours((reg > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(regvis, cs, -1, (255, 140, 0), 3)
            pr = PL.region_prompts(reg, full, neg_map)
            cv2.rectangle(regvis, (pr["box"][0], pr["box"][1]), (pr["box"][2], pr["box"][3]), (255, 255, 0), 2)
            for pp in pr["pos"]: cv2.circle(regvis, tuple(pp), 4, (0, 255, 0), -1)
            for nn in pr["neg"]: cv2.circle(regvis, tuple(nn), 4, (255, 0, 0), -1)
        steps = [("Original", rgb), ("CLIP prior", heat_rgb(clean, pri["clip_ms"])),
                 ("DINOv2 prior", heat_rgb(clean, pri["dino_proto"])),
                 ("Texture prior", heat_rgb(clean, pri["texture"])),
                 ("Fused prior", heat_rgb(clean, full)),
                 ("Region proposals + auto prompts\n(orange region, yellow box, green=+pt, red=-pt)", regvis),
                 (f"Selected SAM2 polygon (IoU {best_iou:.3f})", pred_over(clean, best_pred, gp, gn)),
                 ("GT comparison", gt_over(rgb, gp, gn))]
        fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.ravel()
        for ax, (ttl, im) in zip(axes, steps):
            ax.imshow(im); ax.set_title(ttl, fontsize=10); ax.axis("off")
        fig.suptitle(f"{name} — full fused zero-shot pipeline flow", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(OUT, f"{name}_flow.png"), dpi=90); plt.close()
        print(f"  {name}: compare8 + flow (auto IoU {best_iou:.3f})")


if __name__ == "__main__":
    main()
