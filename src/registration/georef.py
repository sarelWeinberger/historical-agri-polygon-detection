"""
Milestone 1 — semi-automatic georeferencing of the framed ITM map sheets.

The two "Blue Line Ma'on" sheets carry a printed coordinate graticule in Israeli
TM (ITM / EPSG:2039). We:
  1. detect the graticule line pixel positions (thin bright lines),
  2. attach the world coordinate of each labelled line (read from the sheet),
  3. fit a 6-parameter affine pixel->ITM by least squares,
  4. report residuals so a human can judge whether it is trustworthy.

This is deliberately a *provisional* georeference for Milestone 1: the numbers are
recovered from a 1600px cartographic export, not from the full-resolution scans,
so absolute accuracy is limited by the export resolution (~0.13 m/px here). Stage 1
(registration milestone) should redo this against the original orthophotos.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ITM_CRS = "urn:ogc:def:crs:EPSG::2039"  # Israeli Transverse Mercator


def detect_graticule(bgr, panel_x0_frac=0.20, thin_thresh=25, min_frac=0.45):
    """Return (vertical_x[], horizontal_y[]) pixel positions of grid lines."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31).astype(np.float32)
    thin = np.clip(gray.astype(np.float32) - bg, 0, 255)
    x0, x1 = int(panel_x0_frac * w), int(0.995 * w)
    y0, y1 = int(0.01 * h), int(0.99 * h)
    P = thin[y0:y1, x0:x1]
    colsum = (P > thin_thresh).sum(axis=0).astype(float)
    rowsum = (P > thin_thresh).sum(axis=1).astype(float)

    def peaks(v, gap=8):
        thr = min_frac * v.max()
        idx = np.where(v > thr)[0]
        out = []
        if len(idx):
            s = p = idx[0]
            for i in idx[1:]:
                if i - p > gap:
                    out.append(int(round((s + p) / 2))); s = i
                p = i
            out.append(int(round((s + p) / 2)))
        return out

    vx = [g + x0 for g in peaks(colsum)]
    hy = [g + y0 for g in peaks(rowsum)]
    return vx, hy


@dataclass
class Georef:
    gt: tuple            # (a,b,c,d,e,f): X=a*x+b*y+c ; Y=d*x+e*y+f
    crs: str
    rmse_m: float
    max_residual_m: float
    resolution_m: float  # metres per pixel (mean of |a|,|e|)
    n_gcps: int


def fit_affine(gcps):
    """gcps: list of (px, py, X, Y). Least-squares 6-param affine + residuals."""
    px = np.array([[g[0], g[1], 1] for g in gcps], float)
    X = np.array([g[2] for g in gcps], float)
    Y = np.array([g[3] for g in gcps], float)
    cx, *_ = np.linalg.lstsq(px, X, rcond=None)   # a,b,c
    cy, *_ = np.linalg.lstsq(px, Y, rcond=None)   # d,e,f
    gt = (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])
    predX = px @ cx
    predY = px @ cy
    res = np.hypot(predX - X, predY - Y)
    resolution = float((abs(cx[0]) + abs(cy[1])) / 2)
    return Georef(gt=tuple(float(v) for v in gt), crs=ITM_CRS,
                  rmse_m=float(np.sqrt((res ** 2).mean())),
                  max_residual_m=float(res.max()),
                  resolution_m=resolution, n_gcps=len(gcps))


def gcps_from_lines(vx, hy, eastings, northings):
    """Cross the detected lines with the read-off world coords into GCPs.

    eastings[i] is the ITM easting of vertical line vx[i];
    northings[j] is the ITM northing of horizontal line hy[j].
    Produces every (vx[i], hy[j]) intersection as a control point.
    """
    gcps = []
    for xi, E in zip(vx, eastings):
        for yj, N in zip(hy, northings):
            gcps.append((xi, yj, E, N))
    return gcps


def pixel_to_world(points, gt):
    a, b, c, d, e, f = gt
    return [[a * x + b * y + c, d * x + e * y + f] for x, y in points]


# -----------------------------------------------------------------------------
# Control values read from the "Blue Line Ma'on" sheets during Milestone 1.
# Both the 1969 and 1980 sheets share this identical grid pixel layout, verified
# by graticule detection (vx=[516,821,1126,1430], hy=[272,577,881]).
MAON_SHEET_GCPS = {
    "eastings":  [215640, 215680, 215720, 215760],  # for detected vertical lines
    "northings": [591440, 591400, 591360],           # for detected horizontal lines
    "crs": ITM_CRS,
    "note": "provisional; read from 1600px cartographic export, verify in Stage 1",
}


def georeference_maon_sheet(bgr):
    """Full pipeline for a Ma'on ITM sheet -> (Georef, gcps, detected_lines)."""
    vx, hy = detect_graticule(bgr)
    e = MAON_SHEET_GCPS["eastings"]
    n = MAON_SHEET_GCPS["northings"]
    # guard: only pair as many lines as we have labels for
    vx, hy = vx[:len(e)], hy[:len(n)]
    gcps = gcps_from_lines(vx, hy, e[:len(vx)], n[:len(hy)])
    geo = fit_affine(gcps)
    return geo, gcps, (vx, hy)
