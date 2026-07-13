#!/usr/bin/env python3
"""
Render the Experiment-8 deliverables:
  * a 7-panel comparison map per target (GT / GDINO+SAM2 / human SAM2 / CLIP heat /
    CLIP->RP->SAM2 / DINOv2 map / DINOv2->RP->SAM2)
  * a CLIP and a DINOv2 evolution-flow figure
    (Original -> Heatmap -> Region Proposal -> SAM2 Polygon -> GT comparison)
"""
from __future__ import annotations

import json, os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import region_proposal as RP

DATA = "outputs/zeroshot/data"
EXP8 = "outputs/zeroshot/exp8_auto"
GEOM = "outputs/zeroshot/polygons/geometry.json"
OUT = "outputs/zeroshot/exp8_auto/maps_out"
BEST = {"clip": "A_threshold", "dinov2": "D_multithresh"}  # best method per source


def rgb_(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def draw_polys(img, polys, color, thick=3, fill=True):
    vis = img.copy()
    for p in polys:
        pts = np.array(p, np.int32).reshape(-1, 1, 2)
        if fill:
            ov = vis.copy(); cv2.fillPoly(ov, [pts], color); vis = cv2.addWeighted(ov, .35, vis, .65, 0)
        cv2.polylines(vis, [pts], True, color, thick, cv2.LINE_AA)
    return vis


def gt_over(img, gp, gn):
    vis = img.copy()
    for m, c in [(gp, (0, 255, 0)), (gn, (0, 255, 255))]:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, c, 2)
    return vis


def heat_rgb(clean_bgr, heat):
    hn = ((heat - heat.min()) / (heat.max() - heat.min() + 1e-6) * 255).astype(np.uint8)
    hm = cv2.addWeighted(clean_bgr, 0.5, cv2.applyColorMap(hn, cv2.COLORMAP_JET), 0.5, 0)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def metric(results, target, source, method):
    for r in results:
        if r["target"] == target and r["source"] == source and r["method"] == method:
            return r
    return {}


def main():
    os.makedirs(OUT, exist_ok=True)
    geom = json.load(open(GEOM))
    results = json.load(open(os.path.join(EXP8, "results.json")))
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))

    for t in targets:
        name = t["name"]
        clean = cv2.imread(t["clean_path"])
        rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        gp = cv2.imread(t["gt_pos_path"], 0); gn = cv2.imread(t["gt_neg_path"], 0)
        base = gt_over(rgb, gp, gn)
        gm = geom.get(name, {})
        clip_heat = np.load(os.path.join(EXP8, "maps", f"{name}_clip.npy"))
        dino_heat = np.load(os.path.join(EXP8, "maps", f"{name}_dinov2.npy"))
        clip_pred = np.load(os.path.join(EXP8, "maps", f"{name}_clip_{BEST['clip']}_pred.npy"))
        dino_pred = np.load(os.path.join(EXP8, "maps", f"{name}_dinov2_{BEST['dinov2']}_pred.npy"))
        mc = metric(results, name, "clip", BEST["clip"])
        md = metric(results, name, "dinov2", BEST["dinov2"])

        def polypanel(polys, col):
            return draw_polys(base, polys, col)

        panels = [
            ("Ground Truth", gt_over(rgb, gp, gn)),
            (f"GDINO+SAM2 (IoU {gm.get('gdino_sam2',{}).get('mask_iou','-')})",
             polypanel(gm.get("gdino_sam2", {}).get("polygons", []), (60, 60, 255))),
            (f"Prompted SAM2 human (IoU {gm.get('prompted_sam2_box',{}).get('mask_iou','-')})",
             polypanel(gm.get("prompted_sam2_box", {}).get("polygons", []), (255, 255, 255))),
            ("CLIP heatmap", heat_rgb(clean, clip_heat)),
            (f"CLIP->RP->SAM2 (IoU {mc.get('poly_iou','-')})",
             cv2.cvtColor(_pred_overlay(clean, clip_pred, gp, gn), cv2.COLOR_BGR2RGB)),
            ("DINOv2 embedding map", heat_rgb(clean, dino_heat)),
            (f"DINOv2->RP->SAM2 (IoU {md.get('poly_iou','-')})",
             cv2.cvtColor(_pred_overlay(clean, dino_pred, gp, gn), cv2.COLOR_BGR2RGB)),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        axes = axes.ravel()
        for ax in axes: ax.axis("off")
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(im); ax.set_title(title, fontsize=10)
        axes[7].axis("off"); axes[8].axis("off")
        fig.suptitle(f"{name} ({t['year']}) — automatic CLIP/DINOv2 -> region proposal -> SAM2  "
                     f"vs GDINO+SAM2 vs human-prompted SAM2\n"
                     f"green=GT cultivated, cyan=GT hard-neg | pred fill: green=correct, red=FP, blue=on hard-neg",
                     fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUT, f"{name}_compare7.png"), dpi=95); plt.close()

        # evolution flow figures
        for source, heat, pred, met in [("CLIP", clip_heat, clip_pred, mc),
                                        ("DINOv2", dino_heat, dino_pred, md)]:
            hn = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            min_area = 0.004 * gp.shape[0] * gp.shape[1]
            regions = RP.propose(hn, BEST[source.lower()], min_area)[:4]
            regvis = rgb.copy()
            for reg, _c in regions:
                cs, _ = cv2.findContours((reg > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(regvis, cs, -1, (255, 140, 0), 3)
                pr = RP.prompts_from_region(reg, hn)
                cv2.rectangle(regvis, (pr["box"][0], pr["box"][1]), (pr["box"][2], pr["box"][3]), (255, 255, 0), 2)
            steps = [("Original image", rgb),
                     (f"{source} heatmap", heat_rgb(clean, heat)),
                     ("Automatic region proposal\n(orange=region, yellow=auto box)", regvis),
                     (f"SAM2 polygon (IoU {met.get('poly_iou','-')})",
                      cv2.cvtColor(_pred_overlay(clean, pred, gp, gn), cv2.COLOR_BGR2RGB)),
                     ("Ground-truth comparison", gt_over(rgb, gp, gn))]
            fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
            for ax, (title, im) in zip(axes, steps):
                ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
            fig.suptitle(f"{name} — {source} fully-automatic pipeline evolution", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(os.path.join(OUT, f"{name}_{source}_flow.png"), dpi=95); plt.close()
        print(f"  {name}: 7-panel + CLIP/DINOv2 flow")

    print(f"\nmaps -> {OUT}/")


def _pred_overlay(clean_bgr, pred, gp, gn):
    vis = clean_bgr.copy()
    pr = pred > 0; pos = gp > 0; neg = gn > 0
    lay = vis.copy()
    lay[np.logical_and(pr, pos)] = (0, 200, 0)
    lay[np.logical_and(pr, np.logical_not(np.logical_or(pos, neg)))] = (0, 0, 230)
    lay[np.logical_and(pr, neg)] = (230, 60, 0)
    vis = cv2.addWeighted(lay, 0.45, vis, 0.55, 0)
    for m, c in [(gp, (0, 255, 0)), (gn, (0, 255, 255))]:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, c, 2)
    return vis


if __name__ == "__main__":
    main()
