"""
Zero-shot evaluation harness — shared across all model experiments.

Provides:
  * strict single-GPU memory discipline (verify free VRAM, reset/peak stats,
    unload + empty_cache between models);
  * metrics vs the annotation ground truth (cultivated coverage/recall, mask &
    box IoU, false-positive overlap with hard-negative black polygons, general
    false-positive area);
  * rasterisation of boxes / polygons to masks;
  * overlay rendering (GT vs prediction vs false positives);
  * a results recorder that appends one row per experiment.

Nothing here trains anything; ground-truth masks are used only to score.
"""
from __future__ import annotations

import gc
import json
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# GPU memory discipline
# ---------------------------------------------------------------------------
def cuda_free_mb():
    import torch
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    free, total = torch.cuda.mem_get_info()
    return (free / 1e6, total / 1e6)


def reset_peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_mb():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def free_gpu(*objs):
    """Delete model objects, collect, empty cache, return free VRAM (MB)."""
    import torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    free, total = cuda_free_mb()
    return free


@contextmanager
def gpu_session(name: str, min_free_mb: float = 2000):
    """Verify VRAM headroom, track peak, and guarantee cleanup around a model."""
    import torch
    free, total = cuda_free_mb()
    print(f"[{name}] pre-load free VRAM {free:.0f}/{total:.0f} MB")
    if torch.cuda.is_available() and free < min_free_mb:
        raise RuntimeError(f"[{name}] only {free:.0f} MB free (< {min_free_mb}); "
                           "previous model not unloaded?")
    reset_peak()
    t0 = time.time()
    stats = {"name": name}
    try:
        yield stats
    finally:
        stats["peak_vram_mb"] = round(peak_mb(), 1)
        stats["wall_s"] = round(time.time() - t0, 2)
        free_after = free_gpu()
        stats["free_after_mb"] = round(free_after, 1)
        print(f"[{name}] done: peak {stats['peak_vram_mb']} MB, "
              f"{stats['wall_s']}s, free now {free_after:.0f} MB")


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------
def boxes_to_mask(boxes, h, w):
    """boxes: list of [x0,y0,x1,y1] -> filled uint8 mask."""
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1
    return m


def polys_to_mask(polys, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        pts = np.array(p, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
    return m


def mask_to_polygons(mask, min_area=200, simplify_frac=0.004):
    """Binary mask -> list of simplified polygons (list of [x,y])."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = simplify_frac * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) >= 3:
            out.append(ap.tolist())
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _iou(a, b):
    a = a > 0; b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / ua) if ua > 0 else 0.0


def score_prediction(pred_mask, gt_pos, gt_neg, pred_boxes=None, gt_boxes=None):
    """Full metric bundle for one prediction vs the ground truth masks.

    coverage      : recall of cultivated GT     (pred ∩ pos) / pos
    mask_iou      : IoU(pred, cultivated GT)
    fp_on_black   : (pred ∩ neg) / neg          how much hard-negative got hit
    fp_area_frac  : predicted area that is neither pos nor neg, over image area
    precision_pos : (pred ∩ pos) / pred         of predicted area, how much is GT+
    best_box_iou  : best IoU of any pred box vs any GT box (if boxes given)
    """
    pred = pred_mask > 0
    pos = gt_pos > 0
    neg = gt_neg > 0
    pos_a = pos.sum(); neg_a = neg.sum(); pred_a = pred.sum()
    inter_pos = np.logical_and(pred, pos).sum()
    inter_neg = np.logical_and(pred, neg).sum()
    other = np.logical_and(pred, np.logical_not(np.logical_or(pos, neg))).sum()
    out = {
        "coverage": float(inter_pos / pos_a) if pos_a else None,
        "mask_iou": _iou(pred, pos),
        "fp_on_black": float(inter_neg / neg_a) if neg_a else None,
        "fp_area_frac": float(other / pred.size),
        "precision_pos": float(inter_pos / pred_a) if pred_a else None,
        "pred_area_frac": float(pred_a / pred.size),
    }
    if pred_boxes and gt_boxes:
        best = 0.0
        for pb in pred_boxes:
            for gb in gt_boxes:
                best = max(best, box_iou(pb, gb))
        out["best_box_iou"] = round(best, 4)
    return out


def gt_boxes_from_mask(gt_pos):
    """One bounding box per connected cultivated component."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((gt_pos > 0).astype(np.uint8), 8)
    boxes = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area > 100:
            boxes.append([x, y, x + ww, y + hh])
    return boxes


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def overlay(clean_bgr, pred_mask, gt_pos, gt_neg, out_path,
            pred_boxes=None, title=None):
    """GT(green outline) / hard-neg(cyan outline) / prediction fill:
    green where pred correctly hits cultivated, red where FP, blue where on black.
    """
    vis = clean_bgr.copy()
    pred = pred_mask > 0; pos = gt_pos > 0; neg = gt_neg > 0
    tp = np.logical_and(pred, pos)
    fp_black = np.logical_and(pred, neg)
    fp_other = np.logical_and(pred, np.logical_not(np.logical_or(pos, neg)))
    lay = vis.copy()
    lay[tp] = (0, 200, 0)          # correct cultivated (green)
    lay[fp_other] = (0, 0, 230)    # false positive (red)
    lay[fp_black] = (230, 60, 0)   # predicted on hard-negative (blue)
    vis = cv2.addWeighted(lay, 0.45, vis, 0.55, 0)
    # GT outlines
    for c, col in [(gt_pos, (0, 255, 0)), (gt_neg, (255, 255, 0))]:
        cs, _ = cv2.findContours((c > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, col, 2)
    if pred_boxes:
        for b in pred_boxes:
            x0, y0, x1, y1 = [int(v) for v in b]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if title:
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(vis, title[:110], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)
    return out_path


# ---------------------------------------------------------------------------
# Results recorder
# ---------------------------------------------------------------------------
class Results:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.rows = []
        if os.path.exists(path):
            self.rows = json.load(open(path))

    def add(self, **row):
        self.rows.append(row)
        with open(self.path, "w") as f:
            json.dump(self.rows, f, indent=2, ensure_ascii=False)

    def __len__(self):
        return len(self.rows)
