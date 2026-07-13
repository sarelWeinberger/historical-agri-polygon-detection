"""
Automatic region proposal from a [0,1] score/heatmap — NO ground truth involved.

Turns a CLIP/DINOv2 score map into candidate regions and, for each region,
auto-derives SAM 2 prompts (a peak point, several positive points, a rough box).
Four proposal methods (A-D) are provided so they can be compared.

Everything here is label-free: the only inputs are the heatmap and its own peaks.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def smooth(heat, sigma=9):
    return ndi.gaussian_filter(heat.astype(np.float32), sigma)


def _clean(binary, min_area, open_k=5, close_k=9):
    b = binary.astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    regions = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            regions.append((lab == i).astype(np.uint8))
    return regions


def _region_conf(region, heat):
    return float(heat[region > 0].mean())


def method_A(heat, min_area, k=1.0):
    """Global threshold (mean + k*std) + morphology + connected components."""
    h = smooth(heat)
    thr = h.mean() + k * h.std()
    return _clean(h >= thr, min_area)


def method_B(heat, min_area, min_distance=25):
    """Local maxima -> watershed region growing around each peak."""
    h = smooth(heat)
    coords = peak_local_max(h, min_distance=min_distance,
                            threshold_abs=h.mean() + 0.3 * h.std())
    if len(coords) == 0:
        return []
    markers = np.zeros(h.shape, np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    # grow only over above-median area so basins don't flood the whole image
    mask = h >= np.percentile(h, 55)
    labels = watershed(-h, markers, mask=mask)
    regions = []
    for i in range(1, labels.max() + 1):
        r = (labels == i).astype(np.uint8)
        if r.sum() >= min_area:
            regions.append(r)
    return regions


def method_C(heat, min_area, block=51, C=-0.02):
    """Adaptive (local-mean) threshold on the heatmap."""
    h = smooth(heat)
    h8 = (h * 255).astype(np.uint8)
    block = block | 1
    at = cv2.adaptiveThreshold(h8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, block, int(C * 255))
    # keep only above-average heat (adaptive alone also fires on texture)
    at = (at > 0) & (h >= h.mean())
    return _clean(at, min_area)


def method_D(heat, min_area, percentiles=(60, 70, 80, 90)):
    """Multi-threshold: keep candidates from several levels, ranked by confidence."""
    h = smooth(heat)
    seen = []
    cand = []
    for p in percentiles:
        thr = np.percentile(h, p)
        for r in _clean(h >= thr, min_area):
            # dedupe near-identical regions across thresholds
            c = tuple(np.round(ndi.center_of_mass(r), -1))
            if c in seen:
                continue
            seen.append(c)
            cand.append((r, _region_conf(r, h), p))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand  # list of (region, confidence, percentile)


def prompts_from_region(region, heat, n_points=5, box_shrink=0.04):
    """Auto SAM2 prompts for one region — peak point, several points, rough box."""
    ys, xs = np.where(region > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    # rough box, slightly shrunk to avoid the fuzzy heatmap edge
    dw, dh = int((x1 - x0) * box_shrink), int((y1 - y0) * box_shrink)
    box = [x0 + dw, y0 + dh, x1 - dw, y1 - dh]
    # peak = highest-heat pixel inside the region
    hv = heat.copy(); hv[region == 0] = -1
    py, px = np.unravel_index(np.argmax(hv), hv.shape)
    peak = [int(px), int(py)]
    # several positive points = top-heat pixels, spatially spread
    idx = np.argsort(hv[region > 0])[::-1]
    rys, rxs = ys, xs
    order = np.argsort(heat[rys, rxs])[::-1]
    pts, mind = [], max(8, int(0.15 * max(x1 - x0, y1 - y0)))
    for j in order:
        p = [int(rxs[j]), int(rys[j])]
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > mind*mind for q in pts):
            pts.append(p)
        if len(pts) >= n_points:
            break
    if not pts:
        pts = [peak]
    return {"peak": peak, "points": pts, "box": box}


METHODS = {"A_threshold": method_A, "B_localmax": method_B,
           "C_adaptive": method_C, "D_multithresh": method_D}


def propose(heat, method, min_area):
    """Return list of (region_mask, confidence). D returns ranked; others eq-conf."""
    if method == "D_multithresh":
        cand = method_D(heat, min_area)
        return [(r, c) for (r, c, _p) in cand]
    regs = METHODS[method](heat, min_area)
    return [(r, _region_conf(r, smooth(heat))) for r in regs]
