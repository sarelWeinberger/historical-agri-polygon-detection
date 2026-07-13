"""
Milestone 1 — colour-annotation extraction.

The ground truth for this project is *ink drawn over grayscale imagery*:

    yellow  -> cultivated_area      (primary ground truth)
    red     -> cultivated_area      (additional positive example)
    black   -> hard_negative        (looks cultivated but must NOT be detected)

The annotations are thin, un-filled outlines. This module isolates each ink
colour, closes the outline into a ring, fills it into a solid region, and
vectorises it into simplified polygons + a binary mask.

Design choices that matter for a defensible dataset:
  * Saturation is the discriminator for yellow/red (background is grayscale).
  * Black CANNOT be separated from dark terrain by colour alone, so black is
    treated as *candidate* only, is aggressively shape-filtered (closed loops,
    bounded area, not touching the frame), and is always routed to human review.
  * The two framed map exports carry a legend swatch + margins; an ignore mask
    removes the left info panel so the legend is never mistaken for an annotation.
  * The coloured ink is NEVER written into any training raster — we only emit
    masks, polygons and the (separately supplied) original imagery.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# -----------------------------------------------------------------------------
# Colour definitions (OpenCV HSV: H in [0,179], S,V in [0,255])
# -----------------------------------------------------------------------------
CLASS_OF_COLOR = {
    "yellow": "cultivated_area",
    "red": "cultivated_area",
    "black": "hard_negative",
}
REVIEW_REQUIRED = {"black"}  # never auto-trust black


# The imagery underneath is grayscale (R==G==B), so *channel dominance* is a far
# more robust ink detector than fixed HSV saturation cut-offs: any pixel where one
# channel clearly leads the others must be drawn ink, at any brightness. Terrain,
# being neutral, can never satisfy a dominance test.
def yellow_mask(bgr, hsv):
    B, G, R = (bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int))
    H = hsv[..., 0]
    dom = (R > B + 35) & (G > B + 35) & (np.abs(R - G) < 70)   # R,G high, B low
    hue = (H >= 18) & (H <= 42)
    return ((dom & hue).astype(np.uint8)) * 255


def red_mask(bgr, hsv):
    B, G, R = (bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int))
    dom = (R > G + 28) & (R > B + 28)                          # red leads clearly
    return (dom.astype(np.uint8)) * 255


def black_mask(bgr, hsv):
    B, G, R = (bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int))
    S = hsv[..., 1]
    mx = np.maximum(np.maximum(R, G), B)
    # near-pure black ink: dark AND neutral. Kept as *candidate* only; the ring
    # test downstream is what actually separates ink loops from dark terrain.
    return ((mx < 45) & (S < 70)).astype(np.uint8) * 255


MASKERS = {"yellow": yellow_mask, "red": red_mask, "black": black_mask}


@dataclass
class Polygon:
    color: str
    klass: str
    points: list           # [[x,y], ...] in annotation-image pixels
    area_px: float
    perimeter_px: float
    compactness: float     # 4*pi*area / perimeter^2  (1.0 = circle)
    is_closed_loop: bool
    review_required: bool
    holes: list = field(default_factory=list)  # list of point rings


@dataclass
class ExtractionResult:
    image: str
    width: int
    height: int
    polygons: list
    per_color_counts: dict
    ignore_region: Optional[list]  # [x0,y0,x1,y1] excluded (legend/margins)


# -----------------------------------------------------------------------------
def _ignore_box(bgr, is_map_layout: bool):
    """For framed map exports, ignore the left info/legend column."""
    h, w = bgr.shape[:2]
    if is_map_layout:
        return [0, 0, int(0.175 * w), h]  # left panel
    return None


def _apply_ignore(mask, box):
    if box is not None:
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 0
    return mask


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill regions fully enclosed by foreground (flood-fill background trick)."""
    h, w = binary.shape
    pad = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    m = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)          # flood the outside background
    holes = cv2.bitwise_not(ff)[1:-1, 1:-1]    # what the flood couldn't reach
    return cv2.bitwise_or(binary, holes)


def _rings_to_polygons(mask, color, min_area, close_kernel, dilate_iter,
                       simplify_frac, ring_ratio_min):
    """Seal outline gaps -> flood-fill interior -> keep true rings -> vectorise.

    A hand-drawn annotation is a thin *outline*; filling it adds a lot of area
    (fill_area / outline_area is large).  A solid dark terrain blob is already
    filled, so that ratio is ~1.  `ring_ratio_min` is how we reject terrain for
    the black class while accepting genuine outlines for every colour.
    Returns (polygons, filled_mask).
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if dilate_iter:
        closed = cv2.dilate(closed, k, iterations=dilate_iter)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    polys = []
    filled = np.zeros(mask.shape, np.uint8)
    for lbl in range(1, n):
        comp = (labels == lbl).astype(np.uint8) * 255
        solid = _fill_holes(comp)
        if dilate_iter:                        # undo the gap-sealing dilation
            solid = cv2.erode(solid, k, iterations=dilate_iter)
        fill_area = int((solid > 0).sum())
        if fill_area < min_area:
            continue
        # Ring-ness = enclosed area / actual drawn ink inside it. A thin hand-drawn
        # loop encloses many times its own ink (high); a solid dark terrain blob
        # is "ink" everywhere it covers (~1). This is what rejects terrain.
        raw_ink = int(((mask > 0) & (solid > 0)).sum())
        ring_ratio = fill_area / max(raw_ink, 1)
        if ring_ratio < ring_ratio_min:        # solid blob, not an outline
            continue
        cnts, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hier = hier[0]
        outer = max(range(len(cnts)),
                    key=lambda i: cv2.contourArea(cnts[i]) if hier[i][3] == -1 else -1)
        c = cnts[outer]
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        eps = simplify_frac * peri
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        holes = []
        for i in range(len(cnts)):             # real interior holes of this blob
            if hier[i][3] == outer and cv2.contourArea(cnts[i]) > min_area * 0.3:
                ha = cv2.approxPolyDP(cnts[i], eps, True).reshape(-1, 2)
                if len(ha) >= 3:
                    holes.append(ha.tolist())
        compact = float(4 * np.pi * area / (peri * peri + 1e-6))
        polys.append(Polygon(
            color=color, klass=CLASS_OF_COLOR[color],
            points=approx.tolist(), area_px=float(area), perimeter_px=float(peri),
            compactness=round(compact, 3), is_closed_loop=ring_ratio > 2.0,
            review_required=color in REVIEW_REQUIRED, holes=holes,
        ))
        filled = cv2.bitwise_or(filled, solid)
    return polys, filled


def extract(path: str, is_map_layout: bool,
            min_area_frac: float = 3e-4,
            close_kernel: int = 7,
            simplify_frac: float = 0.004) -> tuple[ExtractionResult, dict]:
    """Extract polygons + per-colour filled masks from one annotated image.

    Returns (ExtractionResult, {color: filled_mask uint8}).
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:  # fall back through PIL for exotic encodings
        bgr = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ibox = _ignore_box(bgr, is_map_layout)
    min_area = min_area_frac * h * w

    # adaptive gap-sealing kernel: ~0.6% of the short side, so broken hand-drawn
    # outlines on small crops and large sheets both close.
    close_kernel = max(close_kernel, int(0.006 * min(h, w)) | 1)

    all_polys, masks, counts = [], {}, {}
    for color, fn in MASKERS.items():
        raw = fn(bgr, hsv)
        raw = _apply_ignore(raw, ibox)
        # ring_ratio = enclosed_area / ink. Yellow/red are trusted ink, so a mild
        # threshold just rejects open scribbles. Black must strongly enclose (>>ink)
        # to be told apart from solid dark terrain.
        ring_ratio_min = 4.0 if color == "black" else 1.6
        ma = min_area * (4 if color == "black" else 1)
        dilate_iter = 2 if color == "black" else 1
        polys, filled = _rings_to_polygons(
            raw, color, ma, close_kernel, dilate_iter=dilate_iter,
            simplify_frac=simplify_frac, ring_ratio_min=ring_ratio_min)
        # black: further reject blobs touching the frame (map furniture)
        if color == "black":
            polys, filled = _reject_frame_touching(polys, filled, w, h)
        all_polys.extend(polys)
        masks[color] = filled
        counts[color] = len(polys)

    res = ExtractionResult(
        image=os.path.basename(path), width=w, height=h,
        polygons=all_polys, per_color_counts=counts, ignore_region=ibox)
    return res, masks


def _reject_frame_touching(polys, filled, w, h, margin_frac=0.01):
    m = int(margin_frac * min(w, h)) + 1
    keep, newfilled = [], np.zeros_like(filled)
    for p in polys:
        pts = np.array(p.points)
        touches = (pts[:, 0].min() <= m or pts[:, 1].min() <= m or
                   pts[:, 0].max() >= w - m or pts[:, 1].max() >= h - m)
        if touches:
            continue
        keep.append(p)
        cv2.drawContours(newfilled, [pts.astype(np.int32)], -1, 255, cv2.FILLED)
    return keep, newfilled


# -----------------------------------------------------------------------------
def result_to_dict(res: ExtractionResult) -> dict:
    return {
        "image": res.image, "width": res.width, "height": res.height,
        "ignore_region": res.ignore_region,
        "per_color_counts": res.per_color_counts,
        "polygons": [{
            "color": p.color, "class": p.klass, "points": p.points,
            "holes": p.holes, "area_px": round(p.area_px, 1),
            "perimeter_px": round(p.perimeter_px, 1), "compactness": p.compactness,
            "is_closed_loop": p.is_closed_loop, "review_required": p.review_required,
        } for p in res.polygons],
    }


def debug_overlay(path, res: ExtractionResult, masks: dict, out_png: str):
    """Render extracted fills + polygon outlines over the source for QA."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
    vis = bgr.copy()
    color_bgr = {"yellow": (0, 255, 255), "red": (0, 0, 255), "black": (255, 0, 0)}
    for color, m in masks.items():
        overlay = vis.copy()
        overlay[m > 0] = color_bgr[color]
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    for p in res.polygons:
        pts = np.array(p.points, np.int32)
        cv2.polylines(vis, [pts], True, color_bgr[p.color], 2)
        for hole in p.holes:
            cv2.polylines(vis, [np.array(hole, np.int32)], True, color_bgr[p.color], 1)
    if res.ignore_region:
        x0, y0, x1, y1 = res.ignore_region
        cv2.rectangle(vis, (x0, y0), (x1, y1), (128, 128, 128), 2)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, vis)
