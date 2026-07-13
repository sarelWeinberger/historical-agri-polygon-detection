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

| Phase | Description | State |
|---|---|---|
| **M1** | Scan data, classify images, verify annotations, extract polygons, build masks, viewer + report | ✅ **done** |
| **Zero-shot eval** | Evaluate 7 pretrained models (no training) for cultivation detection on a single 32 GB GPU | ✅ **done** |
| M2 | Baseline semantic segmentation | ⏸ deferred (needs a real labelled set) |
| M3/M4 | Texture detector, temporal analysis, ensemble, final GeoJSON | ⏳ planned |

> **No training has been performed.** The current labelled set is ~one ground-truth field
> per year — insufficient to train a segmenter or trust a holdout.

## Zero-shot pretrained-model evaluation (single 32 GB Blackwell GPU)

Can off-the-shelf models find cultivated fields in 1969/1980 aerial scans *without
training*? Full report: [`outputs/reports/zeroshot_evaluation.md`](outputs/reports/zeroshot_evaluation.md);
comparison table: [`outputs/zeroshot/comparison_table.md`](outputs/zeroshot/comparison_table.md).

**Verdict:** open-vocabulary *detectors* fail (Grounding DINO boxes the whole scene and is
prompt-insensitive; OWLv2 misses cultivation; Florence-2 calls the scenes "airplane/poster"
— out of domain). But two signals **work zero-shot**: **prompted SAM 2** delineates
boundaries at IoU 0.74 (0.86 best) given a human prompt, and **CLIP-L / DINOv2 tile
embeddings** separate cultivated land from look-alike terrain at **AUROC ≈ 0.91–0.92**.
Recommended path: CLIP/DINOv2 region-prior → SAM 2 boundaries → human review; defer training
until more sites are digitised.

Each experiment (`scripts/exp{1..7}_*.py`) loads **one** model at a time with strict VRAM
discipline (verify free → track peak → unload → `empty_cache` → re-verify); peak VRAM never
exceeded 2.4 GB. Florence-2 needs transformers 4.x, isolated in `.venv_flor`.

### Fully-automatic pipeline (exp 8): CLIP/DINOv2 → region proposal → SAM 2

Tests whether a CLIP or DINOv2 score map can **replace the human SAM 2 prompt** (no GT ever
used to build a prompt). Report: [`outputs/reports/exp8_auto_pipeline.md`](outputs/reports/exp8_auto_pipeline.md);
7-panel comparison maps + evolution-flow figures in `outputs/zeroshot/exp8_auto/maps_out/`.

**Result:** the automatic pipeline localises the right fields and beats GDINO+SAM 2
(CLIP→SAM2 IoU 0.23, DINOv2→SAM2 0.21, up to 0.41 on good targets), but stays ~3× below the
human-prompted ceiling (0.74). The **gap is localisation, not segmentation** (same SAM 2:
human box → 0.74, auto box → 0.23). Viable as an automatic *proposal* stage for a
human-review tool; not yet as an autonomous high-precision delineator.

```bash
python scripts/exp8_auto_pipeline.py        # CLIP+DINOv2 maps -> region proposal -> SAM2
python scripts/render_exp8_maps.py          # 7-panel comparison + flow figures
```

```bash
pip install -r requirements-gpu.txt          # Blackwell: torch cu128
python src/evaluation/data.py                 # build ink-free eval targets + GT
for i in 1 2 3 5 6 7; do python scripts/exp${i}_*.py; done
. .venv_flor/bin/activate && python scripts/exp4_florence2.py   # isolated env
python scripts/aggregate_results.py           # comparison table
```

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
