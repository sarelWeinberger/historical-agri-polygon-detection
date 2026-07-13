#!/usr/bin/env python3
"""
Milestone 1 orchestrator — the single entry point that:

  1. scans /home/.../ME/SOURCE and classifies every image,
  2. extracts yellow/red/black polygons from the annotated images,
  3. builds class-indexed + binary masks,
  4. exports GeoJSON in pixel coordinates (all sheets) and ITM/EPSG:2039
     world coordinates (the two georeferenceable map layouts),
  5. writes overlays and a machine-readable dataset summary.

Run:
    python scripts/prepare_dataset.py --src /home/ubuntu/ME/SOURCE \
        --out outputs/milestone1

No model training happens here. Colour ink is never written into an image raster.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.manifest import build_manifest
from src.ingestion.color_annotations import extract, result_to_dict, debug_overlay
from src.vectorization.export import build_masks, build_geojson
from src.registration.georef import georeference_maon_sheet, pixel_to_world, ITM_CRS


def safe_stem(fn: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.splitext(fn)[0])


def process(src: str, out: str) -> dict:
    masks_dir = os.path.join(out, "masks")
    overlays_dir = os.path.join(out, "overlays")
    vectors_dir = os.path.join(out, "vectors")
    poly_dir = os.path.join(out, "polygons")
    for d in (masks_dir, overlays_dir, vectors_dir, poly_dir):
        os.makedirs(d, exist_ok=True)

    manifest = build_manifest(src, os.path.join(out, "manifest.json"))
    summary = {"source_dir": src, "images": [], "totals": {}}
    tot = {"yellow": 0, "red": 0, "black": 0, "cultivated_polys": 0, "hardneg_polys": 0}

    for im in manifest["images"]:
        entry = {"filename": im["filename"], "role": im["role"],
                 "subtype": im["subtype"], "year": im["year"], "notes": im["notes"]}
        if im["role"] != "annotated":
            summary["images"].append(entry)
            continue

        path = im["path"]
        is_map = im["subtype"] == "map_layout"
        res, masks_by_color = extract(path, is_map_layout=is_map)
        stem = safe_stem(im["filename"])

        # masks
        mask_paths, _ = build_masks(res, masks_by_color, masks_dir, stem)
        # overlay
        ov = os.path.join(overlays_dir, f"{stem}_overlay.png")
        debug_overlay(path, res, masks_by_color, ov)
        # polygons json
        with open(os.path.join(poly_dir, f"{stem}.json"), "w") as f:
            json.dump(result_to_dict(res), f, indent=2, ensure_ascii=False)
        # pixel-space geojson
        gj_px = os.path.join(vectors_dir, f"{stem}_pixel.geojson")
        build_geojson(res, gj_px, extra_props={"year": im["year"], "coord_space": "pixel"})

        entry.update({
            "per_color_counts": res.per_color_counts,
            "n_polygons": len(res.polygons),
            "masks": mask_paths, "overlay": ov,
            "geojson_pixel": gj_px,
            "ignore_region": res.ignore_region,
        })

        # world-space geojson for georeferenceable sheets
        if is_map:
            bgr = cv2.imread(path)
            geo, gcps, lines = georeference_maon_sheet(bgr)
            gj_w = os.path.join(vectors_dir, f"{stem}_itm.geojson")
            build_geojson(res, gj_w, geotransform=geo.gt, crs=ITM_CRS,
                          extra_props={"year": im["year"], "coord_space": "ITM_EPSG2039",
                                       "georef_rmse_m": round(geo.rmse_m, 3),
                                       "georef_provisional": True})
            entry["georef"] = {
                "crs": geo.crs, "gt": geo.gt, "rmse_m": round(geo.rmse_m, 4),
                "max_residual_m": round(geo.max_residual_m, 4),
                "resolution_m_per_px": round(geo.resolution_m, 5),
                "n_gcps": geo.n_gcps, "detected_lines": {"vx": lines[0], "hy": lines[1]},
                "geojson_itm": gj_w, "provisional": True,
            }

        tot["yellow"] += res.per_color_counts.get("yellow", 0)
        tot["red"] += res.per_color_counts.get("red", 0)
        tot["black"] += res.per_color_counts.get("black", 0)
        for p in res.polygons:
            if p.klass == "cultivated_area":
                tot["cultivated_polys"] += 1
            elif p.klass == "hard_negative":
                tot["hardneg_polys"] += 1
        summary["images"].append(entry)

    summary["totals"] = tot
    with open(os.path.join(out, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/ubuntu/ME/SOURCE")
    ap.add_argument("--out", default="outputs/milestone1")
    a = ap.parse_args()
    s = process(a.src, a.out)
    print("\n=== Milestone 1 dataset summary ===")
    for im in s["images"]:
        line = f"{im['role']:9s} {im['subtype']:13s} y={im.get('year')}  {im['filename']}"
        if "per_color_counts" in im:
            line += f"  -> {im['per_color_counts']}"
        if "georef" in im:
            line += f"  [ITM rmse={im['georef']['rmse_m']}m res={im['georef']['resolution_m_per_px']}m/px]"
        print(line)
    print("totals:", s["totals"])
    print(f"\nwrote -> {a.out}/  (manifest.json, dataset_summary.json, masks/, overlays/, vectors/, polygons/)")


if __name__ == "__main__":
    main()
