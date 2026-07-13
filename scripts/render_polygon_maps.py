#!/usr/bin/env python3
"""
Render polygon maps for every model on every evaluation target.

Produces, per target:
  * a comparison grid  (one panel per model: clean image + GT outline + model polys + IoU)
  * a combined overlay (all models' polygons on one image, colour-coded, with legend)

Exports GeoJSON per model per target in pixel coordinates, plus ITM/EPSG:2039 for
the two georeferenceable map sheets (using the Milestone-1 graticule transform).
"""
from __future__ import annotations

import json, os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.registration.georef import georeference_maon_sheet, pixel_to_world, ITM_CRS

DATA = "outputs/zeroshot/data"
POLY = "outputs/zeroshot/polygons"
OUT = "outputs/zeroshot/maps"

# stable display order + colours (BGR for cv2, hex for legend)
MODELS = [
    ("grounding_dino",      (60, 60, 255),  "#ff3c3c", "Grounding DINO"),
    ("gdino_sam2",          (0, 140, 255),  "#ff8c00", "GDINO+SAM2"),
    ("owlv2",               (0, 215, 255),  "#ffd700", "OWLv2"),
    ("florence2_grounding", (200, 0, 200),  "#c800c8", "Florence2 ground"),
    ("florence2_refseg",    (255, 0, 150),  "#9600ff", "Florence2 refseg"),
    ("clip",                (0, 230, 0),    "#00e600", "CLIP tiles"),
    ("siglip",              (140, 200, 0),  "#00c88c", "SigLIP tiles"),
    ("dinov2",              (255, 200, 0),  "#00c8ff", "DINOv2 proto*"),
    ("prompted_sam2_box",   (255, 255, 255),"#ffffff", "Prompted SAM2 (human)"),
]


def draw_polys(img, polys, color, thick=2, fill=False):
    vis = img.copy()
    for p in polys:
        pts = np.array(p, np.int32).reshape(-1, 1, 2)
        if fill:
            ov = vis.copy(); cv2.fillPoly(ov, [pts], color); vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        cv2.polylines(vis, [pts], True, color, thick, cv2.LINE_AA)
    return vis


def gt_outline(img, gt_pos, gt_neg):
    vis = img.copy()
    for m, col in [(gt_pos, (0, 255, 0)), (gt_neg, (255, 255, 0))]:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, col, 2)
    return vis


def export_geojson(name, model, polys, geotransform=None):
    feats = []
    for i, p in enumerate(polys):
        ring = [list(map(int, pt)) for pt in p] + [list(map(int, p[0]))]
        coords = pixel_to_world(ring, geotransform) if geotransform else ring
        feats.append({"type": "Feature", "properties": {"model": model, "id": i},
                      "geometry": {"type": "Polygon", "coordinates": [coords]}})
    fc = {"type": "FeatureCollection", "features": feats}
    if geotransform:
        fc["crs"] = {"type": "name", "properties": {"name": ITM_CRS}}
    suffix = "itm" if geotransform else "pixel"
    os.makedirs(os.path.join(OUT, "geojson"), exist_ok=True)
    json.dump(fc, open(os.path.join(OUT, "geojson", f"{name}__{model}_{suffix}.geojson"), "w"),
              indent=2, ensure_ascii=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    geom = json.load(open(os.path.join(POLY, "geometry.json")))
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    tmeta = {t["name"]: t for t in targets}

    for name, models in geom.items():
        t = tmeta[name]
        clean = cv2.imread(t["clean_path"])
        gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
        base = gt_outline(clean, gt_pos, gt_neg)

        # georef transform for the two ITM map sheets
        gt_tf = None
        if t["is_map_layout"]:
            try:
                geo, _, _ = georeference_maon_sheet(clean)
                gt_tf = geo.gt
            except Exception:
                gt_tf = None

        present = [(k, bgr, hexc, lbl) for (k, bgr, hexc, lbl) in MODELS if k in models]

        # ---- comparison grid (matplotlib) ----
        n = len(present); cols = 3; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2))
        axes = np.array(axes).reshape(-1)
        for ax in axes: ax.axis("off")
        for ax, (k, bgr, hexc, lbl) in zip(axes, present):
            d = models[k]
            panel = draw_polys(base, d["polygons"], bgr, thick=3, fill=True)
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            cov = d["coverage"]; cov = f"{cov:.2f}" if isinstance(cov, (int, float)) else "-"
            ax.set_title(f"{lbl}\nIoU={d['mask_iou']}  cov={cov}  polys={len(d['polygons'])}",
                         fontsize=9)
            export_geojson(name, k, d["polygons"])
            if gt_tf and d["polygons"]:
                export_geojson(name, k, d["polygons"], geotransform=gt_tf)
        fig.suptitle(f"{name}  ({t['year']}, {t['subtype']})  —  green=GT cultivated, "
                     f"cyan=GT hard-neg   [* DINOv2 label-informed]", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(OUT, f"{name}_grid.png"), dpi=95)
        plt.close()

        # ---- combined overlay (all models, one image) ----
        combo = base.copy()
        for (k, bgr, hexc, lbl) in present:
            combo = draw_polys(combo, models[k]["polygons"], bgr, thick=2, fill=False)
        cv2.imwrite(os.path.join(OUT, f"{name}_combined.png"), combo)
        # legend
        fig, ax = plt.subplots(figsize=(combo.shape[1] / 120, combo.shape[0] / 120))
        ax.imshow(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)); ax.axis("off")
        handles = [Patch(color="#00ff00", label="GT cultivated"),
                   Patch(color="#00ffff", label="GT hard-neg")] + \
                  [Patch(color=h, label=l) for (k, b, h, l) in present]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=8, frameon=True)
        ax.set_title(f"{name} — all model polygons", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"{name}_combined_legend.png"), dpi=95, bbox_inches="tight")
        plt.close()
        print(f"  {name}: grid + combined ({len(present)} models)")

    print(f"\nmaps -> {OUT}/  (per-target *_grid.png, *_combined_legend.png; geojson/ pixel+ITM)")


if __name__ == "__main__":
    main()
