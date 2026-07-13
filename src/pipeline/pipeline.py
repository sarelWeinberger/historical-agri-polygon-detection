"""
Fusion -> region proposal -> multi-prompt SAM 2 -> label-free polygon selection.

Every scoring signal is inference-time only; GT is never consulted here.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from src.pipeline.priors import norm01


# --------------------------------------------------------------------------- #
# Fusion
def fuse(priors: dict, spec: dict) -> np.ndarray:
    """priors: {name: map[0,1]}. spec: {rule, use:[names], weights:{name:w}, gate:...}"""
    use = spec["use"]
    maps = [norm01(priors[n]) for n in use]
    rule = spec["rule"]
    if rule == "weighted_avg":
        w = np.array([spec.get("weights", {}).get(n, 1.0) for n in use], dtype=np.float64)
        w /= w.sum()
        f = sum(wi * m for wi, m in zip(w, maps))
    elif rule == "geometric_mean":
        f = np.ones_like(maps[0])
        for m in maps: f *= (m + 1e-3)
        f = f ** (1.0 / len(maps))
    elif rule == "rank_avg":
        ranks = [ndi.rank_filter(m, 0, 1) for m in maps]  # placeholder unused
        rk = [(_pixrank(m)) for m in maps]
        f = np.mean(rk, 0)
    elif rule == "min":            # consensus (all must agree)
        f = np.min(maps, 0)
    elif rule == "max":            # union
        f = np.max(maps, 0)
    elif rule == "gate":           # gate[0] gates the product of the rest
        g = maps[0]
        rest = np.mean(maps[1:], 0) if len(maps) > 1 else np.ones_like(g)
        f = g * (0.4 + 0.6 * rest)
    else:
        raise ValueError(rule)
    return norm01(ndi.gaussian_filter(f, spec.get("smooth", 3)))


def _pixrank(m):
    flat = m.ravel(); order = flat.argsort()
    r = np.empty_like(order, np.float64); r[order] = np.linspace(0, 1, len(order))
    return r.reshape(m.shape)


# --------------------------------------------------------------------------- #
# Region proposal (improved set)
def propose(fused, method, min_area, topk=6):
    f = fused
    regions = []
    if method == "multi_thresh":
        for p in (55, 65, 75, 85):
            thr = np.percentile(f, p)
            regions += _cc(f >= thr, min_area, f)
    elif method == "adaptive":
        h8 = (f * 255).astype(np.uint8)
        at = cv2.adaptiveThreshold(h8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, -8)
        regions += _cc((at > 0) & (f >= f.mean()), min_area, f)
    elif method == "watershed":
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        fs = ndi.gaussian_filter(f, 7)
        pk = peak_local_max(fs, min_distance=25, threshold_abs=fs.mean() + 0.3 * fs.std())
        if len(pk):
            mk = np.zeros(f.shape, np.int32)
            for i, (y, x) in enumerate(pk, 1): mk[y, x] = i
            lab = watershed(-fs, mk, mask=fs >= np.percentile(fs, 55))
            for i in range(1, lab.max() + 1):
                r = (lab == i).astype(np.uint8)
                if r.sum() >= min_area: regions.append((r, float(f[r > 0].mean())))
    elif method == "stability":       # regions stable across thresholds
        acc = np.zeros(f.shape, np.float32)
        for p in (55, 62, 70, 78, 86):
            acc += (f >= np.percentile(f, p)).astype(np.float32)
        stable = acc >= 3
        regions += _cc(stable, min_area, f)
    # dedupe by centroid, rank by confidence
    uniq, seen = [], []
    for r, c in sorted(regions, key=lambda z: z[1], reverse=True):
        cen = tuple(np.round(ndi.center_of_mass(r), -1))
        if cen in seen: continue
        seen.append(cen); uniq.append((r, c))
    return uniq[:topk]


def _cc(binary, min_area, f):
    b = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    b = ndi.binary_fill_holes(b).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            r = (lab == i).astype(np.uint8); out.append((r, float(f[r > 0].mean())))
    return out


# --------------------------------------------------------------------------- #
# Automatic prompts from a region + fused map (no GT)
def region_prompts(region, fused, neg_map, n_pos=6, n_neg=4):
    ys, xs = np.where(region > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    dw, dh = int((x1 - x0) * 0.05), int((y1 - y0) * 0.05)
    box = [x0, y0, x1, y1]
    ebox = [x0 + dw, y0 + dh, x1 - dw, y1 - dh]
    hv = fused.copy(); hv[region == 0] = -1
    py, px = np.unravel_index(np.argmax(hv), hv.shape)
    peak = [int(px), int(py)]
    # spread positive points at local maxima inside region
    order = np.argsort(fused[ys, xs])[::-1]
    pos, mind = [], max(8, int(0.15 * max(x1 - x0, y1 - y0)))
    for j in order:
        p = [int(xs[j]), int(ys[j])]
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > mind*mind for q in pos): pos.append(p)
        if len(pos) >= n_pos: break
    if not pos: pos = [peak]
    # negatives: high negative-prior pixels just outside the region
    ring = cv2.dilate(region, np.ones((31, 31), np.uint8)) - region
    nys, nxs = np.where(ring > 0)
    neg = []
    if len(nxs):
        nord = np.argsort(neg_map[nys, nxs])[::-1]
        for j in nord[:200]:
            p = [int(nxs[j]), int(nys[j])]
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > mind*mind for q in neg): neg.append(p)
            if len(neg) >= n_neg: break
    return {"box": box, "ebox": ebox, "peak": peak, "pos": pos, "neg": neg}


# --------------------------------------------------------------------------- #
# Label-free mask selection
def mask_score(mask, fused, neg_map):
    m = mask > 0
    if m.sum() < 10: return -1e9
    fill = float(fused[m].mean())                       # covers high-prior?
    lowprior = float((fused[m] < 0.35).mean())          # spills onto low-prior?
    negcov = float(neg_map[m].mean())                   # covers negative prior?
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return -1e9
    c = max(cnts, key=cv2.contourArea); area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
    compact = 4 * np.pi * area / (peri * peri + 1e-6)
    frag = len(cnts)
    return fill - 0.5 * lowprior - 0.4 * negcov + 0.15 * compact - 0.03 * frag
