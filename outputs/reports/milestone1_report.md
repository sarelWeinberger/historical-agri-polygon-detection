# Milestone 1 — Data Analysis & Annotation Extraction Report

**Project:** Agricultural polygon detection from historical aerial imagery (1969 / 1980)
**Scope of this milestone:** *No model training.* Scan the source data, classify it,
verify the annotation semantics, extract polygons, build masks, provide a viewer, and
document data quality / assumptions / issues before any training begins.

**Source directory:** `/home/ubuntu/ME/SOURCE` (the task brief's `/home/sarel/ME/` did
not exist in this environment; the data was located at `/home/ubuntu/ME/SOURCE`).

---

## 1. What is actually in the dataset

Six image files. The pipeline classifies each automatically (`src/ingestion/manifest.py`):

| File | Dimensions | Ch | Role | Subtype | Year | Notes |
|---|---|---|---|---|---|---|
| `mm188_0567_19091969.jpg` | 33 541 × 32 761 | 1 (gray) | **original** | orthophoto | 1969 | full-res scan, photo date 19.09.1969 |
| `mosaic_mm642_7457_7459_7471_7473_01021980.jpg` | 25 184 × 24 721 | 3 (RGB) | **original** | ortho-mosaic | 1980 | full-res mosaic, photo date 01.02.1980 |
| `source1969.JPE` | 1600 × 1131 | 3 | **annotated** | map_layout | 1969 | framed ITM sheet, 1 yellow polygon |
| `תצלום מ1908 … מסוג שני.jpeg` | 1600 × 1131 | 3 | **annotated** | map_layout | 1980 | framed ITM sheet, 1 yellow polygon, "cultivation type 2" |
| `source1980.JPE` | 1259 × 707 | 3 | **annotated** | example_crop | 1980 | un-framed screenshot: 1 yellow + 2 red + 2 black |
| `דוגמאות פוליגון בתצא 1980.png` | 1213 × 831 | 4 | **annotated** | example_crop | 1980 | un-framed screenshot: 1 yellow + 2 red + 2(?) black |

> Two files were renamed by the user mid-session (`1969-(~1.JPE → source1969.JPE`,
> `(_)~1.JPE → source1980.JPE`) and the two large scans were re-encoded at higher
> quality (same pixel dimensions). The pipeline re-ran cleanly against the new names.

### The two "map layouts" are formal cartographic exports
Both `source1969.JPE` and the `…1908…` sheet are official products of the Israeli Civil
Administration **Central Supervision Unit** ("היחידה המרכזית לפיקוח"), titled
**"קו כחול מעון" (Blue Line — Ma'on)**. Printed metadata:

- **CRS:** *Israel TM* (ITM, **EPSG:2039**).
- **Photo scale** 1:500, **map scale** 1:50 000.
- **Photo dates:** 19.09.**1969** and 01.02.**1980**; **production date** 14.01.2026.
- A **coordinate graticule** (E 215 640–215 760, N 591 360–591 440) — i.e. the
  Ma'on / south-Hebron-hills area — which makes these sheets **georeferenceable**.

---

## 2. Annotation semantics — **verified**, not assumed

The brief's colour convention was confirmed from the legend text and both example crops:

| Ink colour | Meaning | Training class | Trust |
|---|---|---|---|
| **Yellow** | Cultivated land — primary ground truth ("העיבודים לפי שנת 1980") | `cultivated_area` | high |
| **Red** | Additional positive cultivated example | `cultivated_area` | high |
| **Black** | Hard negative — looks cultivated but is **not** | `hard_negative` | **low (see §4)** |

Visual confirmation: in both crops the yellow field shows clear plough/terrace **row
texture**; the red loops enclose obviously cultivated rectangular fields; the black loops
sit on rocky / eroded / terraced slopes that mimic cultivation. **The assumption holds.**

---

## 3. Extraction pipeline & results

`scripts/prepare_dataset.py` runs the whole chain. Ink is isolated by **channel
dominance** (robust because the underlying imagery is grayscale, so any pixel where one
channel leads the others must be drawn ink), the thin outline is gap-sealed, flood-filled
into a solid region, and vectorised with hole preservation. A **ring-ness test**
(enclosed-area ÷ ink-pixels) separates true drawn loops from solid features.

**Extracted totals:** yellow **4**, red **4**, black **4** → **8 cultivated** polygons
and **4 hard-negative** candidates across the 4 annotated images.

| Image | yellow | red | black | Correct? |
|---|---|---|---|---|
| `source1969.JPE` (map) | 1 | 0 | 0 | ✅ sheet has only the yellow field |
| `…1908…` (map, 1980) | 1 | 0 | 0 | ✅ same |
| `source1980.JPE` (crop) | 1 | 2 | 2 | ✅ all 5 match the drawing exactly |
| `דוגמאות…1980.png` (crop) | 1 | 2 | 2 | ⚠️ yellow+red ✅; **black wrong** (see §4) |

Per-image overlays, class masks, binary masks, polygon JSON and GeoJSON are in
`outputs/milestone1/{overlays,masks,polygons,vectors}/`. Browse everything in
**`outputs/milestone1/viewer.html`**.

### Georeferencing (provisional)
`src/registration/georef.py` detects the graticule, attaches the ITM coordinates read
off each labelled line, and fits a 6-parameter affine pixel→ITM:

- **RMSE 0.047 m**, max residual < 0.05 m, **0.131 m/px**, 12 GCPs — a clean, axis-aligned fit.
- The 1969 yellow field reprojects to **E 215 655–215 735, N 591 344–591 462**, area
  **≈ 6 290 m² (0.63 ha)** — a plausible field size, inside the printed grid. ✅
- Emitted as `*_itm.geojson` (EPSG:2039). **Marked provisional**: it is recovered from a
  1600-px export, not the full-resolution scan; Stage 1 must redo it properly.

---

## 4. Data-quality findings, issues & inconsistencies

**A. The labelled dataset is tiny — this is the dominant risk.**
Effectively **one** georeferenced ground-truth field per year (the single Ma'on site),
plus ~4 red positives and ~4 black-negative candidates on two screenshots. This is far
too little to train the Stage-2 segmentation model. **We need many more digitised sites
before training is meaningful.**

**B. Black hard-negatives are only partly recoverable by colour.**
Confirmed empirically: on the *light* crop (`source1980.JPE`) black extraction is 2/2
correct; on the *dark* PNG the detector **missed the real black rectangle** (its interior
is dark terrain, so the ring test cannot tell ink from background) and **produced 2 false
blobs** from dark terrain. → All black is flagged `review_required=True` and must be
digitised/validated by hand. Do **not** feed raw black extractions to training unchecked.

**C. Resolution mismatch.** Annotations were digitised on ~1600-px cartographic exports;
the originals are ~33 000 px. Polygon boundaries are therefore coarse relative to the raw
imagery and will need refinement (SAM boundary stage) once mapped onto full-res scans.

**D. Registration gaps.**
- The two **original** scans are plain JPEGs with **no world file / embedded CRS**.
- Only the two **map sheets** carry georeferencing (the graticule); the two **crops** carry none.
- We therefore cannot yet place the crop/red/black polygons in world coordinates — they
  exist only in pixel space until the crops are matched to an original (Stage 1 feature matching).

**E. Annotation inconsistencies.**
- The **1969** sheet's legend still reads *"cultivations per year 1980"* — a template
  label that was not updated for the 1969 product.
- The 1980 sheet's **filename says "1908"** — a typo; the sheet's printed photo date is
  01.02.1980 (corrected via a logged override in the manifest).
- The two crops are un-framed screenshots at **unknown scale/orientation**.

**F. Missing pairs / missing data.**
- No **clean (un-annotated)** copy of the crops — the ink is baked into the only image we
  have at that resolution, so a pristine "original vs annotation" pair does not exist for them.
- No **vector ground truth** (GeoJSON/SHP) was supplied — only rasterised ink, which we
  had to re-vectorise (a lossy round-trip).
- No labels at all on the two **full-resolution originals**.

---

## 5. Assumptions made (please verify)

1. Colour map yellow/red = positive, black = hard-negative (**verified visually — holds**).
2. `source1980.JPE` and `דוגמאות…1980.png` are crops of the **1980** imagery (inferred
   from RGB tone + the "1980" filename; the geographic location is *not yet confirmed*
   against the mosaic).
3. The 1969 and 1980 map sheets cover the **same** ground extent (identical graticule
   pixel layout strongly supports this).
4. ITM (EPSG:2039) is the correct CRS (printed on the sheets).
5. The graticule labels were read correctly (215 640–215 760 E; 591 360–591 440 N) —
   worth a second human check before relying on the world coordinates.

---

## 6. Recommended next steps (before Milestone 2 training)

1. **Acquire far more labelled sites.** One field/year cannot train a segmenter. Digitise
   additional yellow/red/black polygons across both scans (ideally as vector files).
2. **Prefer vector ground truth.** Ask the data owner for the original GeoJSON/SHP behind
   these sheets to avoid the raster→vector round-trip.
3. **Register the originals (Stage 1).** Give the two scans a real geotransform (from the
   sheets' ITM grid + feature matching), then place *all* polygons in EPSG:2039.
4. **Locate the crops** within the 1980 mosaic by feature matching, so their red/black
   examples become georeferenced training data.
5. **Hand-validate every black** hard-negative in the viewer; re-digitise the ones the
   colour detector cannot recover over dark terrain.
6. Only then proceed to **Milestone 2** (baseline segmentation) with a geographic
   train/val/holdout split (never random tiles).

---

## 7. Deliverables produced by this milestone

- `outputs/milestone1/manifest.json` — every image classified with colour stats & notes.
- `outputs/milestone1/dataset_summary.json` — machine-readable extraction summary.
- `outputs/milestone1/overlays/` — extraction overlays (QA).
- `outputs/milestone1/masks/` — class-indexed + binary cultivated / hard-negative masks.
- `outputs/milestone1/polygons/` — per-image polygon JSON (area, compactness, review flag).
- `outputs/milestone1/vectors/` — GeoJSON in pixel space (all) + ITM/EPSG:2039 (map sheets).
- `outputs/milestone1/viewer.html` — self-contained annotation viewer / validation UI.
- This report.

**Status: Milestone 1 complete. Do not start training until §6 (esp. #1) is addressed.**
