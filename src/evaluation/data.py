"""
Zero-shot evaluation — data preparation.

For every annotated image we build an *ink-free* evaluation target so the models
never see annotation strokes as evidence:

  clean_rgb   : the source with yellow/red/black outlines inpainted away
  gt_pos_mask : union of yellow + red filled polygons  (cultivated ground truth)
  gt_neg_mask : black filled polygons                  (hard negatives)
  polygons    : the vector ground truth (from Milestone 1 extraction)

The polygons are EVALUATION GROUND TRUTH ONLY — never used to train anything.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.ingestion.color_annotations import extract, MASKERS, _apply_ignore, _ignore_box
from src.ingestion.manifest import build_manifest


@dataclass
class EvalTarget:
    name: str
    filename: str
    year: int
    subtype: str
    clean_path: str          # ink-free RGB
    gt_pos_path: str         # cultivated mask (uint8 0/255)
    gt_neg_path: str         # hard-negative mask
    polygons: list           # list of dicts (color, class, points, holes)
    width: int
    height: int
    is_map_layout: bool
    meta: dict = field(default_factory=dict)


def _combined_ink_mask(bgr, is_map_layout):
    """Union of all colour ink (raw), dilated, for inpainting."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ibox = _ignore_box(bgr, is_map_layout)
    ink = np.zeros(bgr.shape[:2], np.uint8)
    for color, fn in MASKERS.items():
        m = fn(bgr, hsv)
        m = _apply_ignore(m, ibox)
        ink |= (m > 0).astype(np.uint8) * 255
    # widen to cover anti-aliased halo around each stroke
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(ink, k, iterations=1)


def prepare(src_dir: str, out_dir: str, include_maps: bool = True) -> list[EvalTarget]:
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_manifest(src_dir, os.path.join(out_dir, "eval_manifest.json"))
    targets = []
    for im in manifest["images"]:
        if im["role"] != "annotated":
            continue
        is_map = im["subtype"] == "map_layout"
        if is_map and not include_maps:
            continue
        stem = "".join(c if c.isalnum() or c in "._-" else "_"
                       for c in os.path.splitext(im["filename"])[0])
        bgr = cv2.imread(im["path"], cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]

        # ink-free clean image (inpaint strokes only; interiors preserved)
        ink = _combined_ink_mask(bgr, is_map)
        clean = cv2.inpaint(bgr, ink, 5, cv2.INPAINT_TELEA)
        # for map layouts, also blank the left info/legend panel to neutral
        ibox = _ignore_box(bgr, is_map)
        if ibox:
            x0, y0, x1, y1 = ibox
            clean[y0:y1, x0:x1] = int(clean.mean())
        clean_path = os.path.join(out_dir, f"{stem}_clean.png")
        cv2.imwrite(clean_path, clean)

        # GT masks from Milestone 1 extraction
        res, masks_by_color = extract(im["path"], is_map_layout=is_map)
        pos = np.zeros((h, w), np.uint8)
        for c in ("yellow", "red"):
            pos |= (masks_by_color.get(c, np.zeros((h, w), np.uint8)) > 0).astype(np.uint8)
        neg = (masks_by_color.get("black", np.zeros((h, w), np.uint8)) > 0).astype(np.uint8)
        pos_path = os.path.join(out_dir, f"{stem}_gt_pos.png")
        neg_path = os.path.join(out_dir, f"{stem}_gt_neg.png")
        cv2.imwrite(pos_path, pos * 255)
        cv2.imwrite(neg_path, neg * 255)

        polys = [{"color": p.color, "class": p.klass, "points": p.points,
                  "holes": p.holes, "area_px": p.area_px} for p in res.polygons]

        targets.append(EvalTarget(
            name=stem, filename=im["filename"], year=im["year"] or -1,
            subtype=im["subtype"], clean_path=clean_path,
            gt_pos_path=pos_path, gt_neg_path=neg_path, polygons=polys,
            width=w, height=h, is_map_layout=is_map,
            meta={"per_color_counts": res.per_color_counts,
                  "pos_px": int(pos.sum()), "neg_px": int(neg.sum())},
        ))

    index = [t.__dict__ for t in targets]
    with open(os.path.join(out_dir, "eval_targets.json"), "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return targets


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/ME/SOURCE"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/zeroshot/data"
    ts = prepare(src, out)
    for t in ts:
        print(f"{t.name:20s} y={t.year} {t.subtype:12s} {t.width}x{t.height} "
              f"pos={t.meta['pos_px']} neg={t.meta['neg_px']} polys={len(t.polygons)}")
    print(f"\n{len(ts)} eval targets -> {out}")
