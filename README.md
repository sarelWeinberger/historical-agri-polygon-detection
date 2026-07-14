# Agricultural Polygon Detection from Historical Aerial Imagery (1969 / 1980)

Detecting and delineating **cultivated agricultural land** in historical black-and-white
orthophotos, treating cultivated fields as a **geometric and textural phenomenon** (row
orientation, boundaries, texture) rather than a simple object class.

This repository is a **mature research project**, not a prototype. It contains an end-to-end,
**fully training-free** system plus the evaluation evidence behind every design decision:

- **Dataset preparation** — scan/classify the imagery, verify the colour-annotation
  semantics, extract polygons, build masks, and recover ITM georeferencing.
- **Zero-shot benchmarking** — a controlled evaluation of 7 pretrained vision models
  (Grounding DINO, GDINO+SAM 2, prompted SAM 2, Florence-2, OWLv2, CLIP/SigLIP, DINOv2).
- **Automatic polygon generation** — turn a CLIP/DINOv2 score map into automatic SAM 2
  prompts (Experiment 8), showing the bottleneck is *localization*, not segmentation.
- **Optimized fusion pipeline** — the current production recommendation:
  CLIP multi-scale + texture → fusion → region proposal → multi-prompt SAM 2 →
  label-free polygon selection.
- **Evaluation framework** — shared metrics (poly/mask IoU, coverage, boundary-F1, area
  ratio, fragments, false-positive overlap), overlays, comparison maps, and per-round tracking.

**Implemented today:** everything above — dataset prep, benchmarking, automatic polygon
generation, the optimized zero-shot fusion pipeline, and the evaluation framework.
**No model has been trained or fine-tuned.** All ground-truth polygons are used *only* to
score results after inference — never to generate prompts.

**Future work (not implemented):** a lightweight *learned* localization head on frozen
CLIP/DINOv2 embeddings (the one high-value step beyond zero-shot), and multi-year temporal
change analysis. Both require a larger labelled set than currently exists.

## Status

| Phase | Description | Status |
|---|---|---|
| Milestone 1 | Dataset analysis, annotation extraction, georeferencing | ✅ Complete |
| Zero-shot benchmark | Evaluation of pretrained models | ✅ Complete |
| Experiment 8 | Automatic CLIP/DINOv2 → SAM 2 pipeline | ✅ Complete |
| Optimized fused pipeline | CLIP + texture → proposal → SAM 2 | ✅ Complete |
| Lightweight learned localization head | Planned | ⏳ |
| Multi-year temporal analysis | Planned | ⏳ |

> Single-GPU throughout: NVIDIA RTX PRO 4500 Blackwell, 32 GB, CUDA 13.2. One major model
> is loaded at a time with strict VRAM discipline (verify free → track peak → unload →
> `empty_cache` → re-verify); peak VRAM never exceeded ~2.4 GB. Florence-2 requires
> transformers 4.x and is isolated in `.venv_flor`.

---

## Milestone 1 — dataset analysis & annotation extraction

Report: [`outputs/reports/milestone1_report.md`](outputs/reports/milestone1_report.md) ·
viewer: `outputs/milestone1/viewer.html`

1. **Scans** the source directory and classifies each image as `original` orthophoto,
   `map_layout` (framed ITM cartographic export), or `example_crop`.
2. **Verifies the annotation semantics** — yellow/red → `cultivated_area`,
   black → `hard_negative` (confirmed from the legend and example crops).
3. **Extracts polygons** by channel-dominance ink isolation + gap-seal + flood-fill +
   ring-ness filtering (robust to duller reds and neutral blacks; rejects dark terrain).
4. **Builds masks** (class-indexed 0/1/2 + binary) and **GeoJSON** (pixel always;
   ITM / EPSG:2039 for the two georeferenceable map sheets, graticule RMSE ≈ 5 cm).
5. **Reports** dataset quality, registration issues, annotation inconsistencies, missing pairs.

The coloured annotation ink is **never** written into any image raster — only masks and
vectors are emitted. The labelled set is small (≈ one ground-truth field per year), which is
why the project is deliberately zero-shot and defers supervised training.

---

## Zero-shot benchmark of pretrained models

Report: [`outputs/reports/zeroshot_evaluation.md`](outputs/reports/zeroshot_evaluation.md) ·
table: [`outputs/zeroshot/comparison_table.md`](outputs/zeroshot/comparison_table.md)

Can off-the-shelf models find cultivated fields in 1969/1980 scans *without training*?

- **Open-vocabulary detectors fail.** Grounding DINO boxes the whole scene and is
  prompt-insensitive; OWLv2 misses cultivation while over-firing on terrain; Florence-2
  labels the scenes "airplane/poster/animal" — out of domain.
- **Two signals work zero-shot.** **Prompted SAM 2** delineates boundaries at IoU 0.74
  (0.86 best) given a good prompt, and **CLIP-L / DINOv2 tile embeddings** separate cultivated
  land from look-alike terrain at **AUROC ≈ 0.91–0.92**.

Conclusion: segmentation is solved by SAM 2; the missing capability is *localization*.

---

## Experiment 8 — automatic CLIP/DINOv2 → SAM 2

Report: [`outputs/reports/exp8_auto_pipeline.md`](outputs/reports/exp8_auto_pipeline.md) ·
maps: `outputs/zeroshot/exp8_auto/maps_out/`

Replace the human SAM 2 prompt with an automatic one derived from a CLIP or DINOv2 score
map (no GT ever used to build a prompt). It localizes the correct fields and beats
GDINO+SAM 2 (CLIP→SAM 2 IoU 0.23, DINOv2→SAM 2 0.21, up to 0.41 on good targets), but stays
~3× below the human ceiling. Same SAM 2, human box → 0.74 vs auto box → 0.23, so the **gap
is localization, not segmentation** — the finding that motivated the optimized pipeline.

---

## Optimized Zero-Shot Production Pipeline

Report: [`outputs/reports/fused_pipeline_report.md`](outputs/reports/fused_pipeline_report.md) ·
maps: [`outputs/pipeline/maps/`](outputs/pipeline/maps/) ·
config: [`configs/production_pipeline.yaml`](configs/production_pipeline.yaml)

```
Historical aerial image
        ↓
CLIP multi-scale semantic prior
        +
classical texture prior
        ↓
fusion  (weighted average, 2 : 1)
        ↓
multi-threshold region proposal
        ↓
automatic SAM 2 prompts  (box + eroded-box + positive/negative points)
        ↓
multi-prompt SAM 2
        ↓
label-free polygon selection
        ↓
final polygons
```

**DINOv2 was evaluated extensively and removed from the production configuration.** Its
frozen embeddings separate the classes well (AUROC ≈ 0.91), but as an automatic *localizer*
its label-free prior is bimodal — strong on one map sheet (IoU 0.79), collapsing to 0.00 on
the crops — so every CLIP+DINOv2 fusion (0.16–0.24) scored *below* CLIP alone (0.35). It is
retained in the benchmark for evidence but excluded from the production pipeline because it
**reduced overall robustness**.

### Final optimization results

| Pipeline | Mean IoU | Best IoU | Coverage | Notes |
|---|---|---|---|---|
| Grounding DINO + SAM 2 | 0.20 | — | 0.97 | Whole-scene localization (not useful) |
| CLIP → SAM 2 (Exp 8) | 0.23 | 0.41 | 0.31 | First automatic pipeline |
| **Optimized fused pipeline** | **0.365** | **0.71** | **0.69** | **Production recommendation** |
| Prompted SAM 2 | 0.743 | 0.86 | — | Human-assisted ceiling |

The optimized pipeline improves mean IoU by **≈ 61 %** over the original automatic pipeline
(0.227 → 0.365), and — just as important — removes the collapses: worst-case IoU rises from
0.08 to **0.237**, coverage from 0.31 to 0.69, at ≈ 2 fragments per target. It reaches
**≈ 49 % of the human-prompted SAM 2 ceiling** (0.365 vs 0.743).

### Why the improvement happened

The gain came from the localization/proposal stack, **not** from segmentation:

- **multi-scale CLIP** with historical-aerial prompt ensembles (sharper, better-grounded prior);
- a **classical texture / row prior** (structure-tensor coherence + Gabor + FFT periodicity)
  that rescues the worst-case target where CLIP alone is weak;
- **multi-threshold region proposal** (best of the proposal methods tested);
- **multiple SAM 2 prompt strategies** per region (box, eroded box, positive and negative points);
- **label-free mask selection** — pick the mask that best fills the fused prior and avoids the
  negative prior, rather than trusting SAM 2's own confidence.

**SAM 2 itself was never the bottleneck.** Given a correct prompt it already reaches IoU 0.86
on a single field; the limiting factor is producing a precise prompt automatically.

---

## Recommended Production Configuration

```
CLIP multi-scale
        +
Texture prior            (weighted-average fusion, 2 : 1)
        ↓
Multi-threshold connected components
        ↓
Automatic box + positive / negative prompts
        ↓
SAM 2
        ↓
Label-free polygon ranking
```

Full parameters in [`configs/production_pipeline.yaml`](configs/production_pipeline.yaml).
This is the **currently recommended deployment for a human-review workflow**: it reliably
locates the fields and emits clean, low-fragment candidate polygons that a reviewer confirms
or nudges, rather than drawing from scratch. It is **not** yet accurate enough for fully
autonomous mapping.

```bash
pip install -r requirements-gpu.txt            # Blackwell: torch cu128
python src/evaluation/data.py                  # build ink-free eval targets + GT
python scripts/pipeline_compute_priors.py      # cache CLIP / DINOv2 / texture priors
python scripts/pipeline_optimize.py            # fusion x proposal sweep (dev-set tuning)
python scripts/pipeline_render_maps.py         # 8-panel comparison + full-flow maps
```

---

## Conclusions

- **CLIP is the dominant localization signal** — the backbone of the pipeline.
- **Texture improves robustness** — it lifts the worst-case target rather than the mean.
- **DINOv2 did not improve the production pipeline** despite promising embeddings; it is
  unstable as an automatic localizer and was dropped from the fusion.
- **SAM 2 is responsible only for boundary delineation** — it is not the bottleneck.
- **The remaining bottleneck is dense localization precision**, especially on low-contrast /
  dark crops where CLIP's tile-scale separability drops (AUROC 0.27 vs 0.93 on the lighter crop).
- The pipeline is **suitable as an automatic proposal system for human review, but not yet
  for fully autonomous mapping** (mean IoU 0.365 ≈ 49 % of the human ceiling).

Scientific scope, kept explicit: the **zero-shot experiments** measure off-the-shelf models
as-is; the **optimized fused pipeline** is a *heuristic* fusion tuned by bounded dev-set
selection on the 4 existing targets (labelled as such — no generalization is claimed); and a
**future supervised localization head** is the recommended next step once more sites are
digitised. The evidence (AUROC ≈ 0.91–0.92 embedding separability, but automatic IoU capped at
~0.365) shows the limitation is localization precision, not boundary segmentation.

---

## Reports & artifacts

| Phase | Report | Maps / artifacts |
|---|---|---|
| Milestone 1 | [`milestone1_report.md`](outputs/reports/milestone1_report.md) | `outputs/milestone1/` (masks, vectors, `viewer.html`) |
| Zero-shot benchmark | [`zeroshot_evaluation.md`](outputs/reports/zeroshot_evaluation.md) | [`comparison_table.md`](outputs/zeroshot/comparison_table.md), `outputs/zeroshot/maps/` |
| Experiment 8 | [`exp8_auto_pipeline.md`](outputs/reports/exp8_auto_pipeline.md) | `outputs/zeroshot/exp8_auto/maps_out/` |
| **Optimized fused pipeline** | [`fused_pipeline_report.md`](outputs/reports/fused_pipeline_report.md) | [`outputs/pipeline/maps/`](outputs/pipeline/maps/), [`configs/production_pipeline.yaml`](configs/production_pipeline.yaml) |

---

## Layout

```
configs/          production_pipeline.yaml   winning zero-shot configuration
                  milestone1.yaml            dataset-extraction parameters
src/
  ingestion/      manifest.py, color_annotations.py   scan/classify + ink→polygons→masks
  registration/   georef.py                 graticule detection + affine pixel→ITM
  vectorization/  export.py                 masks + GeoJSON (pixel & ITM)
  visualization/  viewer.py                 HTML annotation viewer
  evaluation/     harness.py, data.py, sam2_helper.py, prompts.py, region_proposal.py
  pipeline/       priors.py, pipeline.py     CLIP/DINOv2/texture priors, fusion, proposal, selection
scripts/
  prepare_dataset.py                         Milestone 1 entry point
  exp{1..8}_*.py, render_exp8_maps.py        zero-shot benchmark + Experiment 8
  pipeline_compute_priors.py                 cache priors (one model at a time)
  pipeline_optimize.py, pipeline_round2.py   fusion × proposal optimization
  pipeline_save_variants.py, pipeline_render_maps.py   per-variant preds + final maps
  aggregate_results.py                       master comparison table
outputs/
  milestone1/     manifest, masks, overlays, polygons, vectors, viewer.html
  zeroshot/       per-experiment results, comparison_table, maps, exp8_auto/
  pipeline/       baseline (frozen), rounds (metrics), maps (8-panel + flow)
  reports/        milestone1 / zeroshot_evaluation / exp8_auto_pipeline / fused_pipeline
```

## Engineering notes

- Python 3.11 target (this environment ran 3.12; noted in the M1 report). GPU stack:
  `requirements-gpu.txt` (torch cu128 for Blackwell, transformers 5.x, SAM 2 native).
- Milestone-1-only stack: `requirements.txt` (OpenCV, Pillow, NumPy, scikit-image, Shapely,
  Rasterio, GeoPandas, PyProj).
- **Offline after install; no imagery is uploaded to any external service.** GT polygons are
  used only to score, never to prompt. CRS for georeferenced products: **Israeli TM, EPSG:2039**.
- Reproducible: priors are cached; every optimization round records the full metric suite;
  raw `.npy` prior/prediction caches are gitignored (regenerable) while maps, results JSON,
  reports, and configs are tracked.

## Data & licensing

The historical imagery and Israeli Civil Administration map sheets are **not** committed to
this repository (see `.gitignore`) — only code, small masks/vectors/overlays, maps, and
reports are tracked. Verify redistribution rights before sharing the source imagery.
