# Agricultural Polygon Detection from Historical Aerial Imagery (1969 / 1980)

Production-oriented pipeline for detecting and delineating **cultivated agricultural
land** in historical black-and-white orthophotos, treating cultivated fields as a
**geometric and textural phenomenon** (row orientation, boundaries, texture, temporal
consistency) rather than a simple object class.

The full design follows a four-stage architecture — **Registration → Candidate
generation (semantic segmentation + Grounding DINO + SAM2 + classical texture) →
Fusion → Multi-year temporal analysis**. This repository currently implements
**Milestone 1** (data analysis & annotation extraction); later milestones add training,
detectors and temporal analysis.

## Status

| Milestone | Description | State |
|---|---|---|
| **1** | Scan data, classify images, verify annotations, extract polygons, build masks, viewer + report | ✅ **done** |
| 2 | Baseline semantic segmentation + polygonisation + evaluation | ⏳ planned |
| 3 | Grounding DINO + SAM2 + classical texture detector + hard-negative mining | ⏳ planned |
| 4 | Registration + temporal analysis + ensemble fusion + final GeoJSON | ⏳ planned |

> **Do not start model training yet** — see the Milestone 1 report's "next steps".
> The current labelled set is one ground-truth field per year, which is insufficient
> to train a segmenter.

## Milestone 1 — what it does

1. **Scans** the source directory and classifies each image as `original` orthophoto,
   `map_layout` (framed ITM cartographic export), or `example_crop`.
2. **Verifies the annotation semantics** — yellow/red → `cultivated_area`,
   black → `hard_negative` (confirmed from the legend and example crops).
3. **Extracts polygons** by channel-dominance ink isolation + gap-seal + flood-fill +
   ring-ness filtering (robust to duller reds and neutral blacks; rejects dark terrain).
4. **Builds masks** (class-indexed 0/1/2 + binary) and **GeoJSON** (pixel space always;
   ITM / EPSG:2039 for the two georeferenceable map sheets, RMSE ≈ 5 cm).
5. **Renders** per-image overlays and a self-contained **`viewer.html`** validation UI.
6. **Reports** dataset quality, registration issues, annotation inconsistencies and
   missing pairs: `outputs/reports/milestone1_report.md`.

The coloured annotation ink is **never** written into any image raster — only masks and
vectors are emitted.

## Quick start

```bash
# Ubuntu: system Python 3.12 here (target is 3.11); venv module may need installing
sudo apt-get install -y python3-venv        # if `python3 -m venv` fails
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Run the whole Milestone 1 pipeline
python scripts/prepare_dataset.py --src /home/ubuntu/ME/SOURCE --out outputs/milestone1
python -m src.visualization.viewer outputs/milestone1     # -> outputs/milestone1/viewer.html
```

Open `outputs/milestone1/viewer.html` in a browser to review every extraction.

## Layout

```
configs/                      Hydra/YAML configs (per-stage; grows with later milestones)
data/{raw,annotations,masks,tiles,splits}
src/
  ingestion/     manifest.py           scan + classify images
                 color_annotations.py  ink -> polygons -> masks (yellow/red/black)
  registration/  georef.py             graticule detection + affine pixel->ITM
  vectorization/ export.py             masks + GeoJSON (pixel & ITM) export
  visualization/ viewer.py             HTML annotation viewer
  segmentation/ grounding/ sam/ texture/ fusion/ temporal/ evaluation/   (Milestones 2-4)
scripts/         prepare_dataset.py    Milestone 1 entry point
outputs/
  milestone1/    manifest, masks, overlays, polygons, vectors, viewer.html
  reports/       milestone1_report.md
```

## Engineering notes

- Python 3.11 target (this environment ran 3.12.3 — noted in the report).
- Core libs: OpenCV, Pillow, NumPy, scikit-image, Shapely, Rasterio, GeoPandas, PyProj.
- Offline after dependency install; **no imagery is uploaded to any external service**.
- Every extraction is reproducible from the manifest (paths + `sha16` content hashes).
- CRS for the georeferenced products: **Israeli TM, EPSG:2039**.

## Data & licensing

The historical imagery and Israeli Civil Administration map sheets are **not** committed
to this repository (see `.gitignore`) — only code, small masks/vectors/overlays and
reports are tracked. Verify redistribution rights before sharing the source imagery.
