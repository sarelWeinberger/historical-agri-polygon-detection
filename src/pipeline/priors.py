"""
Zero-shot region priors for the cultivated-area pipeline (NO training).

  * CLIP multi-scale semantic prior (prompt ensembles + overlap blending + local contrast)
  * classical texture / row prior (structure-tensor coherence, Gabor energy,
    FFT periodicity, entropy) — favours parallel rows / coherent field texture,
    suppresses chaotic rocky texture
  * DINOv2 helpers live alongside but need the GPU model (see compute script)

All maps are returned as dense float32 in [0,1] at full image resolution.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
# Prompt ensembles specialised for historical grayscale aerial imagery
CLIP_POS = [
    "historical aerial photograph of cultivated agricultural land",
    "historical black and white aerial image of a plowed field",
    "agricultural terrace visible from above",
    "field with parallel cultivation rows",
    "worked agricultural soil in an old aerial photograph",
    "cultivated plot in a grayscale aerial survey",
    "plowed agricultural field", "terraced farmland",
]
CLIP_NEG = [
    "rocky uncultivated slope in an aerial photograph",
    "natural erosion pattern", "barren hillside", "exposed rock",
    "dirt road", "scan artifact", "shadow", "uncultivated terraced terrain",
    "rocky terrain", "natural bushland",
]


def tile_grid(H, W, size, overlap):
    step = max(1, int(round(size * (1 - overlap))))
    xs = list(range(0, max(1, W - size + 1), step)) or [0]
    ys = list(range(0, max(1, H - size + 1), step)) or [0]
    if W > size and xs[-1] != W - size: xs.append(W - size)
    if H > size and ys[-1] != H - size: ys.append(H - size)
    return [(x, y) for y in ys for x in xs]


def _cos_window(size):
    w = np.hanning(size)
    return np.clip(np.outer(w, w), 1e-3, 1).astype(np.float32)


def blend_scale(H, W, coords, values, size):
    """Accumulate per-tile scalar `values` into a dense map with a cosine window."""
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win = _cos_window(size)
    for (x, y), v in zip(coords, values):
        acc[y:y+size, x:x+size] += v * win
        wsum[y:y+size, x:x+size] += win
    return acc / np.maximum(wsum, 1e-6)


def norm01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def local_contrast(heat, sigma_frac=0.12):
    """High-pass: emphasise local peaks relative to a large neighbourhood."""
    s = max(4, int(sigma_frac * min(heat.shape)))
    bg = ndi.gaussian_filter(heat, s)
    hp = heat - bg
    return norm01(hp)


# --------------------------------------------------------------------------- #
# Classical texture / row prior
def structure_coherence(gray, sigma=6):
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = ndi.gaussian_filter(gx * gx, sigma)
    Jyy = ndi.gaussian_filter(gy * gy, sigma)
    Jxy = ndi.gaussian_filter(gx * gy, sigma)
    tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    coh = tmp / (Jxx + Jyy + 1e-6)          # 1 = strongly oriented (rows)
    return np.clip(coh, 0, 1)


def gabor_energy(gray, freqs=(0.12, 0.2, 0.3), n_theta=8, ksize=21):
    g = gray.astype(np.float32) / 255.0
    best = np.zeros_like(g)
    responses = []
    for f in freqs:
        for t in range(n_theta):
            theta = np.pi * t / n_theta
            lam = 1.0 / f
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lam, 0.5, 0, cv2.CV_32F)
            kern -= kern.mean()
            r = cv2.filter2D(g, cv2.CV_32F, kern)
            responses.append(np.abs(r))
    responses = np.stack(responses, 0)
    best = responses.max(0)
    # orientation selectivity: peak-to-mean of directional responses
    return best


def fft_periodicity(gray, size, overlap=0.5):
    """Per-tile FFT peak-to-mean (rows/terraces are periodic -> high)."""
    H, W = gray.shape
    coords = tile_grid(H, W, size, overlap)
    vals = []
    win = np.hanning(size)[:, None] * np.hanning(size)[None, :]
    for (x, y) in coords:
        t = gray[y:y+size, x:x+size].astype(np.float32)
        t = (t - t.mean()) * win
        F = np.abs(np.fft.fftshift(np.fft.fft2(t)))
        c = size // 2
        F[c-2:c+3, c-2:c+3] = 0                      # kill DC / low freq
        vals.append(float(F.max() / (F.mean() + 1e-6)))
    return blend_scale(H, W, coords, vals, size)


def local_entropy(gray, disk=9):
    from skimage.filters.rank import entropy
    from skimage.morphology import disk as _disk
    return entropy((gray).astype(np.uint8), _disk(disk)).astype(np.float32)


def texture_row_prior(gray):
    """Dense [0,1] prior favouring coherent parallel rows, suppressing chaos."""
    coh = structure_coherence(gray, sigma=6)
    gab = norm01(gabor_energy(gray))
    per = norm01(fft_periodicity(gray, size=96, overlap=0.5))
    ent = norm01(local_entropy(gray, disk=9))
    var = norm01(cv2.GaussianBlur((gray.astype(np.float32)) ** 2, (0, 0), 9)
                 - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 9) ** 2)
    # coherence is the key row signal; gabor+periodicity reinforce; entropy suppresses
    core = norm01(coh) * (0.5 + 0.5 * gab) * (0.5 + 0.5 * per)
    core = core * (1.0 - 0.35 * ent)               # damp chaotic/rocky
    core = core * (0.3 + 0.7 * var)                # require some structure
    return norm01(ndi.gaussian_filter(core, 5))
