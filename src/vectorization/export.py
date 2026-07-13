"""
Milestone 1 — turn extracted polygons into the standard deliverables:
  * class-indexed segmentation mask (0 bg / 1 cultivated / 2 hard_negative)
  * binary masks per super-class
  * GeoJSON in pixel coordinates (always) and, when a geotransform is supplied,
    a second GeoJSON in Israeli TM (EPSG:2039) world coordinates.

Colour ink is never written into a raster that could be used as an image input;
only masks / vectors are emitted here.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

# class-index palette for the segmentation mask
CLASS_INDEX = {"background": 0, "cultivated_area": 1, "hard_negative": 2,
               "uncertain_boundary": 3}
PALETTE = {0: (0, 0, 0), 1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 165, 255)}


def _poly_np(points):
    return np.array(points, np.int32).reshape(-1, 1, 2)


def build_masks(res, masks_by_color, out_dir, stem):
    """Write class-indexed + binary masks. Returns dict of written paths."""
    os.makedirs(out_dir, exist_ok=True)
    h, w = res.height, res.width
    class_mask = np.zeros((h, w), np.uint8)
    # paint hard-negative first, then cultivated on top (cultivated wins overlaps)
    if "black" in masks_by_color:
        class_mask[masks_by_color["black"] > 0] = CLASS_INDEX["hard_negative"]
    cult = np.zeros((h, w), np.uint8)
    for c in ("yellow", "red"):
        if c in masks_by_color:
            cult |= (masks_by_color[c] > 0).astype(np.uint8)
    class_mask[cult > 0] = CLASS_INDEX["cultivated_area"]

    paths = {}
    p = os.path.join(out_dir, f"{stem}_mask_class.png")
    cv2.imwrite(p, class_mask); paths["class_mask"] = p
    # colour preview of the class mask
    prev = np.zeros((h, w, 3), np.uint8)
    for idx, col in PALETTE.items():
        prev[class_mask == idx] = col
    p = os.path.join(out_dir, f"{stem}_mask_class_preview.png")
    cv2.imwrite(p, prev); paths["class_mask_preview"] = p
    # binary masks
    p = os.path.join(out_dir, f"{stem}_mask_cultivated.png")
    cv2.imwrite(p, (cult * 255).astype(np.uint8)); paths["cultivated"] = p
    if "black" in masks_by_color:
        p = os.path.join(out_dir, f"{stem}_mask_hardneg.png")
        cv2.imwrite(p, masks_by_color["black"]); paths["hard_negative"] = p
    return paths, class_mask


def _ring_world(ring, gt):
    """Apply affine geotransform gt=(a,b,c,d,e,f): X=a*x+b*y+c, Y=d*x+e*y+f."""
    a, b, c, d, e, f = gt
    return [[a * x + b * y + c, d * x + e * y + f] for x, y in ring]


def build_geojson(res, out_path, geotransform=None, crs=None, extra_props=None):
    """Write a FeatureCollection. Pixel coords unless a geotransform is given."""
    feats = []
    for i, p in enumerate(res.polygons):
        outer = p.points + [p.points[0]]  # close ring
        rings = [outer] + [h + [h[0]] for h in p.holes]
        if geotransform is not None:
            rings = [_ring_world(r, geotransform) for r in rings]
        props = {
            "id": i, "color": p.color, "class": p.klass,
            "area_px": round(p.area_px, 1), "perimeter_px": round(p.perimeter_px, 1),
            "compactness": p.compactness, "is_closed_loop": p.is_closed_loop,
            "review_required": p.review_required,
            "detection_source": "color_annotation_extraction",
            "source_image": res.image,
        }
        if extra_props:
            props.update(extra_props)
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": rings},
                      "properties": props})
    fc = {"type": "FeatureCollection", "features": feats}
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": crs}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)
    return out_path
