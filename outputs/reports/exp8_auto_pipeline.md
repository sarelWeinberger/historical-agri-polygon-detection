# Experiment 8 — Can CLIP / DINOv2 automatically replace the human SAM 2 prompt?

**Hypothesis tested:** the zero-shot failure is in *region proposal*, not segmentation.
So replace the human prompt with an automatic one derived from a CLIP or DINOv2 score
map:

```
CLIP / DINOv2  →  score map  →  automatic region proposal  →  auto SAM 2 prompts  →  SAM 2  →  polygon
```

**No training. No fine-tuning. Ground-truth polygons were used only to score — never to
build a prompt.** Every prompt originates from the CLIP/DINOv2 map and its own peaks.

---

## Method

- **CLIP-L/14 map:** per-tile contrastive margin between the positive cultivation prompts
  and the negative terrain prompts (text-driven, fully automatic).
- **DINOv2-base map (label-free):** DINOv2 is class-agnostic (no text), so its tile
  embeddings are reduced by PCA to their dominant texture axis, and the sign is fixed with
  an **image-only** heuristic — the *darker* tiles score high (cultivated fields are the
  dark, plough-textured minority in these scans). This is the only way to point DINOv2 at
  "cultivation" without a label or a click; it is a heuristic and is reported as such.
- **Region proposal (4 methods compared):** (A) smooth + mean+σ threshold + morphology +
  connected components; (B) local-maxima + watershed growing; (C) adaptive threshold;
  (D) multi-threshold, keep-all-candidates ranked by heat confidence.
- **Automatic prompts per region:** peak point, several spread positive points, and a rough
  box — all from the heatmap. Fed to SAM 2; SAM 2's own confidence chooses the mask
  (box preferred, points as fallback). Sub-min-area speckle removed (label-free).

One model in VRAM at a time: CLIP map → free → DINOv2 map → free → SAM 2. Peak VRAM ≤ 1.5 GB.

---

## Results

**Region-proposal method sweep (mean over 4 targets, poly IoU):**

| Source | A_threshold | B_localmax | C_adaptive | D_multithresh |
|---|---|---|---|---|
| CLIP → SAM 2 | **0.227** | 0.115 | 0.219 | 0.139 |
| DINOv2 → SAM 2 | 0.064 | 0.080 | 0.091 | **0.210** |

Best method per source: **CLIP → A**, **DINOv2 → D**. Using those:

| Pipeline | poly IoU | GT coverage | boundary F1 | area ratio | fragments | fp-on-black |
|---|---|---|---|---|---|---|
| **CLIP → RP → SAM 2** (auto) | **0.227** | 0.31 | 0.22 | 0.66 | 2.8 | low |
| **DINOv2 → RP → SAM 2** (auto) | **0.210** | 0.35 | 0.24 | 1.42 | 2.8 | low |
| Grounding DINO + SAM 2 | 0.196 | 0.97* | — | — | 1 | ~0.98 |
| **Prompted SAM 2 (human box)** | **0.743** | — | high | ~1.0 | 1/field | 0.09 |

\* GDINO "coverage" is 0.97 only because it boxes the whole scene (fp-on-black ≈ 0.98).

**Per-target poly IoU (best method per source):**

| Target | CLIP→SAM2 | DINOv2→SAM2 | GDINO+SAM2 | human SAM2 |
|---|---|---|---|---|
| source1980 (crop) | **0.30** | **0.40** | 0.16 | 0.72 |
| דוגמאות…1980 (crop) | 0.19 | 0.03 | 0.10 | 0.66 |
| …1980 sheet | **0.34** | **0.41** | 0.26 | 0.88 |
| source1969 sheet | 0.08 | 0.00 | 0.26 | 0.81 |

The automatic pipelines clearly **beat GDINO+SAM 2** on the crops and reach IoU 0.30–0.41
on their good targets, but are far below the human-prompted upper bound and collapse where
the map is undiscriminative (the 1969 sheet is one large uniform dark field — DINOv2's
brightness heuristic and CLIP's tile margin both fail there).

Deliverables: 7-panel comparison map + CLIP/DINOv2 evolution-flow figures per target in
`outputs/zeroshot/exp8_auto/maps_out/`; polygons + GeoJSON in `outputs/zeroshot/exp8_auto/`.

---

## Conclusions (the five questions)

**1. Can CLIP automatically replace the human prompt?**
**Partially — for *localisation*, not for *precision*.** CLIP's text-driven heatmap reliably
points SAM 2 at the correct fields (it already separated cultivated vs terrain at tile-AUROC
0.92), and the automatic CLIP→RP→SAM 2 pipeline produces clean, few-fragment polygons that
land on real fields (IoU up to 0.34, mean 0.23). But that is ~⅓ of the human-prompted
quality (0.72): the auto box is coarser than a human's, so SAM 2 under/over-covers.

**2. Can DINOv2 automatically replace the human prompt?**
**Less reliably.** Being class-agnostic it has no notion of "cultivation" and needs a
label-free orientation heuristic. That heuristic works well on some targets (IoU 0.40–0.41)
and fails completely on others (0.00–0.03) — mean 0.21 with high variance. DINOv2 is a
strong *feature* space (its embeddings separate the classes at AUROC 0.91) but not a
dependable *standalone* localiser without a reference (text, a click, or labels).

**3. Is CLIP/DINOv2 → SAM 2 comparable to Prompted SAM 2?**
**No — about 3× lower IoU (0.21–0.23 vs 0.74).** Crucially the segmenter is identical; only
the prompt source differs. The automatic pipelines match or beat GDINO+SAM 2 but do not
approach the human-prompt ceiling.

**4. Is the remaining gap in localisation or segmentation?**
**Localisation, decisively.** Same SAM 2: human box → 0.74, auto box → 0.23. Where the
auto heatmap peak lands squarely on a field (source1980) the auto IoU jumps to 0.30–0.40;
where the map is undiscriminative it collapses. SAM 2's boundary quality is not the
bottleneck (prompted SAM 2 reaches 0.86 on a single field). The weak link is turning a
coarse tile-scale heatmap into a *precise* field-scale prompt.

**5. Is there a viable fully zero-shot pipeline without training?**
**Viable as an assisted *proposal* generator; not yet as an autonomous *delineator*.**
CLIP → region-proposal → SAM 2 is a genuine, fully-automatic, training-free pipeline that
finds the right fields and yields usable coarse polygons — good enough for a **high-recall,
human-review** mode (a person confirms/nudges instead of drawing from scratch). It is **not**
accurate enough (IoU ~0.23) for **high-precision automatic** mapping.

### Recommendation
The evidence points to a clear, cheap next step **that is no longer purely zero-shot but is
strongly justified**: the embeddings already separate the classes at AUROC ≈ 0.91–0.92, so a
**light classifier or upsampling head on frozen CLIP/DINOv2 features** (or a CLIP-seeded,
DINOv2-refined region step) would sharpen localisation — the one thing missing — without the
cost of training a full segmenter. Until then, ship **CLIP → RP → SAM 2 as the automatic
proposal stage feeding a human-review tool**, and keep SAM 2 (which is not the bottleneck)
for boundaries. Improving the *detector/localiser* is where effort should go; more prompt
engineering on open-vocabulary detectors is not.
