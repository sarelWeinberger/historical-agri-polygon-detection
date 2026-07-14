# Trying `geoai` (opengeos/geoai-py) on the historical imagery — verdict

**Ask:** evaluate whether [`geoai`](https://github.com/opengeos/geoai) (`geoai-py 0.41.1`)
offers a zero-shot capability that beats our fused CLIP+texture → SAM 2 pipeline for
cultivated-field polygons on the 1969/1980 grayscale aerial imagery. **No training.**

## What geoai offers (relevant zero-shot tools)

`geoai` is a broad geospatial-AI toolkit (data download, dataset prep, **training**,
inference, visualization, QGIS). Its zero-shot / pretrained inference entry points include:

| geoai tool | Underlying model | Relevance here |
|---|---|---|
| **`CLIPSegmentation`** | CLIPSeg (`CIDAS/clipseg-rd64-refined`) — text→dense mask | **Most relevant** — a *dense* text localizer, the exact thing our tile-scale CLIP prior lacks |
| `GroundedSAM` | GroundingDINO + SAM | Same family we already benchmarked (GDINO boxes the whole scene) |
| `moondream_detect`, `rfdetr_detect` | Moondream VLM, RF-DETR | Natural-image detectors (out-of-domain, like OWLv2/Florence-2) |
| `DINOv3Segmenter`, `TimmSegmentationModel`, `train_*` | supervised seg/detection | Require **training** — out of scope for this zero-shot phase |

The genuinely new zero-shot capability is **CLIPSeg dense text segmentation**, so that is what
we tested rigorously.

## Result — CLIPSeg is out-of-domain and underperforms our pipeline

**Direct `CLIPSegmentation.segment_image` (default threshold 0.4)** on the ink-free eval crops:

| Target | best IoU (any agri prompt) |
|---|---|
| source1969 sheet | 0.045 |
| source1980 crop | 0.099 |
| דוגמאות…1980 crop | 0.000 |
| …1980 sheet | 0.374 |
| **mean** | **0.130** |

Most prompts return an **empty mask**. Diagnosing the raw sigmoid probability (prompt
ensemble, max over 5 agricultural phrases) shows why:

- **CLIPSeg barely activates** on grayscale historical imagery — raw probability maxes out at
  **0.03–0.08** across the whole image, so any fixed threshold produces empty/garbage masks
  (oracle-threshold IoU ≈ 0).
- The *relative* ranking carries **weak** signal: AUROC vs background **0.76** (0.61–0.86),
  AUROC vs hard-negative **0.76** on the two crops.
- Our own **CLIP tile prior scores AUROC 0.93** (cultivated vs hard-neg on source1980) — a much
  cleaner localizer. The CLIPSeg overlay (`outputs/geoai/*_clipseg_heat.png`) confirms it:
  it catches the centre field but fires heavily on rocky terrain and misses the bottom-left field.

**Comparison:**

| Method | mean IoU | note |
|---|---|---|
| geoai CLIPSeg (default) | 0.130 | mostly empty masks |
| geoai CLIPSeg (oracle threshold) | ~0.00 | probabilities too low to threshold |
| CLIP → SAM 2 (Exp 8) | 0.23 | our first automatic pipeline |
| **Fused CLIP+texture → SAM 2 (ours)** | **0.365** | production recommendation |
| Prompted SAM 2 | 0.743 | human ceiling |

## Verdict

`geoai` is a capable library, but for **this** task it does not help zero-shot:

- **CLIPSeg** (its flagship zero-shot text segmenter) is **out-of-domain** on 1969/1980
  grayscale aerial scans — near-inactive output, and its weak relative signal (AUROC 0.76) is
  **below our existing multi-scale CLIP prior (0.93)**. It underperforms our fused pipeline
  (0.130 vs 0.365) and would not improve the fusion.
- **`GroundedSAM`** is the GroundingDINO+SAM combination we already showed fails here
  (whole-scene boxes); **`moondream`/`rfdetr`** are natural-image detectors expected to be
  out-of-domain like OWLv2/Florence-2; the strong parts of geoai (`DINOv3Segmenter`,
  `train_segmentation_*`, torchgeo) all **require training**, which this phase excludes.

**Recommendation:** keep the fused CLIP+texture → SAM 2 pipeline as the zero-shot production
system. `geoai` becomes relevant **later** — its `train_segmentation_*` / DINOv3 / torchgeo
tooling is a good fit for the planned *supervised* localization head once a labelled dataset
exists (it would also handle tiling, georeferencing, and QGIS export out of the box). This
matches the standing conclusion that the remaining bottleneck is localization precision, which
needs a small amount of training rather than another zero-shot model.

*(geoai isolated in `.venv_geoai`; scripts: `scripts/try_geoai_clipseg.py`,
`scripts/clipseg_prior_eval.py`.)*
