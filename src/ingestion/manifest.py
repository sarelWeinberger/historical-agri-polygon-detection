"""
Milestone 1 — directory scanning, image classification and manifest building.

Scans a source directory of historical aerial imagery and produces a structured
manifest that separates:
  * original orthophotos / mosaics  (grayscale, very large, no colour ink)
  * annotated products               (colour polygons drawn over imagery)
      - "map_layout"  : a framed cartographic export with a coordinate graticule
                        (georeferenceable) and a legend / info panel.
      - "example_crop": an un-framed screenshot with coloured polygons only.

Classification is deliberately heuristic and fully transparent — every decision
is recorded in the manifest so a human can audit it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # historical scans are huge; we trust local files

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".bmp"}

# Date tokens frequently embedded in the Israeli CA file names, e.g.
# mm188_0567_19091969.jpg  -> 19-09-1969 ;  ..._01021980.jpg -> 01-02-1980
import re
_DATE_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def sha256_head(path: str, nbytes: int = 8 << 20) -> str:
    """Hash of the first `nbytes` (enough to fingerprint without reading GBs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


def guess_dates(name: str) -> dict:
    ddmmyyyy = None
    m = _DATE_RE.search(name)
    if m:
        d, mo, y = m.groups()
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
            ddmmyyyy = f"{y}-{mo}-{d}"
    years = sorted({int(y) for y in _YEAR_RE.findall(name)})
    return {"photo_date": ddmmyyyy, "years_in_name": years}


@dataclass
class ImageRecord:
    filename: str
    path: str
    size_bytes: int
    width: int
    height: int
    channels: int
    dtype: str
    sha16: str
    photo_date: Optional[str]
    years_in_name: list
    # colour statistics (computed on a downscaled proxy)
    sat_fraction: float          # fraction of pixels with meaningful saturation
    yellow_px: int
    red_px: int
    has_white_margin: bool       # large flat near-white border -> map layout / legend
    has_left_info_panel: bool    # tall white legend column -> framed ITM map export
    # derived classification
    role: str = "unknown"        # original | annotated
    subtype: str = "unknown"     # orthophoto | map_layout | example_crop
    year: Optional[int] = None
    notes: list = field(default_factory=list)


def _proxy(path: str, long_side: int = 1600):
    """Load a downscaled BGR proxy quickly (JPEG draft mode for big files)."""
    im = Image.open(path)
    w, h = im.size
    mode = im.mode
    im.draft(mode, (long_side, long_side))
    im.thumbnail((long_side, long_side))
    arr = np.array(im.convert("RGB"))[:, :, ::-1].copy()  # -> BGR
    return arr, (w, h), mode


def _colour_stats(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 40) & (V > 40)
    yellow = (H >= 20) & (H <= 38) & (S > 90) & (V > 120)
    red = (((H <= 8) | (H >= 170)) & (S > 110) & (V > 90))
    # near-white flat border -> cartographic margin
    white = (V > 235) & (S < 20)
    hh, ww = bgr.shape[:2]
    border = np.zeros((hh, ww), bool)
    b = max(2, int(0.02 * min(hh, ww)))
    border[:b] = border[-b:] = border[:, :b] = border[:, -b:] = True
    has_white_margin = white[border].mean() > 0.15 or white.mean() > 0.10
    # A framed cartographic sheet carries a tall white info/legend panel down its
    # left edge. That single feature separates the two ITM map exports from the
    # un-framed example crops far more reliably than a generic white-margin test.
    left = white[:, : int(0.16 * ww)]
    left_panel = left.mean() > 0.45
    return dict(
        sat_fraction=float(sat.mean()),
        yellow_px=int(yellow.sum()),
        red_px=int(red.sum()),
        has_white_margin=bool(has_white_margin),
        has_left_info_panel=bool(left_panel),
    )


def _true_shape(path: str):
    """True dimensions/channels without loading the whole raster into RAM twice."""
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
    ch = {"L": 1, "P": 1, "RGB": 3, "RGBA": 4, "I;16": 1}.get(mode, len(mode))
    dtype = "uint16" if mode == "I;16" else "uint8"
    return w, h, ch, dtype, mode


def scan_directory(src: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
            continue
        w, h, ch, dtype, mode = _true_shape(p)
        bgr, _, _ = _proxy(p)
        cs = _colour_stats(bgr)
        dt = guess_dates(fn)
        rec = ImageRecord(
            filename=fn, path=p, size_bytes=os.path.getsize(p),
            width=w, height=h, channels=ch, dtype=dtype, sha16=sha256_head(p),
            photo_date=dt["photo_date"], years_in_name=dt["years_in_name"],
            **cs,
        )
        _classify(rec)
        records.append(rec)
    _pair(records)
    return records


# Known filename/content corrections, applied transparently and logged in notes.
# The 1980 ITM sheet's file name contains the typo "1908" (its printed photo
# date is 01.02.1980). Verified visually during Milestone 1.
OVERRIDES = {
    "תצלום מ1908 - פוליגון של עיבוד מסוג שני.jpeg": {
        "year": 1980,
        "photo_date": "1980-02-01",
        "note": "filename token '1908' is a typo; sheet prints photo date 01.02.1980",
    },
}


def _classify(r: ImageRecord) -> None:
    has_ink = (r.yellow_px + r.red_px) > 300
    megapix = (r.width * r.height) / 1e6
    if has_ink or r.has_left_info_panel:
        r.role = "annotated"
        if r.has_left_info_panel and megapix < 6:
            r.subtype = "map_layout"
            r.notes.append("framed ITM cartographic export; carries coordinate graticule")
        else:
            r.subtype = "example_crop"
            r.notes.append("un-framed annotated screenshot (no graticule)")
    else:
        r.role = "original"
        r.subtype = "orthophoto"
        if megapix > 100:
            r.notes.append("full-resolution historical scan")
    # year
    if r.photo_date:
        r.year = int(r.photo_date[:4])
    elif r.years_in_name:
        r.year = r.years_in_name[0]
    # plausibility: aerial photography of this region did not exist in 1908
    if r.year is not None and r.year < 1935:
        r.notes.append(f"implausible year {r.year} parsed from name; likely a typo")
    # apply verified overrides
    ov = OVERRIDES.get(r.filename)
    if ov:
        r.year = ov["year"]
        r.photo_date = ov["photo_date"]
        r.notes.append("OVERRIDE: " + ov["note"])


def _pair(records: list[ImageRecord]) -> None:
    """Attach cross-references between originals and annotations of same year."""
    originals = {r.year: r for r in records if r.role == "original" and r.year}
    for r in records:
        if r.role == "annotated" and r.year in originals:
            r.notes.append(f"same-year original: {originals[r.year].filename}")


def build_manifest(src: str, out_json: str) -> dict:
    records = scan_directory(src)
    manifest = {
        "source_dir": src,
        "n_images": len(records),
        "images": [asdict(r) for r in records],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/ME/SOURCE"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/milestone1/manifest.json"
    m = build_manifest(src, out)
    for im in m["images"]:
        print(f"{im['role']:9s} {im['subtype']:13s} y={im['year']} "
              f"{im['width']}x{im['height']} ch{im['channels']} "
              f"Y={im['yellow_px']} R={im['red_px']}  {im['filename']}")
    print(f"\nmanifest -> {out}")
