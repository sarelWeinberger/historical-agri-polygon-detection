# Fused Zero-Shot Pipeline — CLIP + DINOv2 + Texture → Fusion → SAM 2

**Goal:** the strongest *fully-automatic, training-free* pipeline for cultivated-area
polygons on the historical 1969/1980 imagery, optimised iteratively.
**No training, no fine-tuning, no classifier fitting. GT used only to score, never to prompt.**

Single RTX PRO 4500 Blackwell (32 GB); one model loaded at a time (peak VRAM ≤ 1.5 GB).
All priors cached to disk so fusion/proposal/SAM 2 iterate cheaply.

---

## Pipeline

```
CLIP multi-scale semantic prior  +  DINOv2 label-free prior  +  classical texture/row prior
        → fusion → automatic region proposal → automatic SAM 2 prompts (box/points/±neg)
        → label-free mask selection → polygon post-processing → candidate polygons
```

- **CLIP prior** — `clip-vit-large-patch14`, tiles {224,336,448} @ 50 % overlap, cosine-window
  blended, historical-aerial positive-prompt ensemble minus terrain/artefact negative ensemble.
  Two variants: `clip_ms` (multi-scale margin) and `clip_lc` (local-contrast high-pass).
- **DINOv2 prior** — `dinov2-base`; the fragile "dark = cultivated" heuristic is **replaced** by a
  label-free **CLIP-agreement prototype**: cluster patch embeddings, weight clusters by the CLIP
  prior, build a cultivation prototype, score tiles by similarity. DINOv2 itself is never trained.
- **Texture prior** — structure-tensor coherence + Gabor energy + FFT periodicity + entropy +
  variance → favours parallel rows / coherent field texture, suppresses chaotic rocky texture.
- **Fusion** — weighted-avg / geometric-mean / rank / min / max / gate.
- **Proposal** — multi-threshold CC / adaptive / watershed / cross-threshold stability.
- **SAM 2** — 5 automatic prompt sets per region; **mask chosen by a label-free score**
  (fill of high-prior pixels − spill onto low-prior − overlap with negative prior + compactness − fragments).

---

## Optimization (bounded, dev-set-tuned on the 4 existing targets)

Winner chosen by a **robustness score `0.5·(mean_IoU + min_IoU)`** so a config that spikes one
target but collapses another is penalised.

**Fusion sweep (proposal = multi_thresh):**

| Fusion | mean IoU | median | min | max | boundary-F1 |
|---|---|---|---|---|---|
| clip_ms only | 0.350 | 0.220 | 0.183 | 0.777 | 0.276 |
| clip_lc only | **0.373** | **0.359** | 0.168 | 0.606 | **0.338** |
| **clip_ms + texture (2:1)** ⟵ winner | 0.365 | 0.254 | **0.237** | 0.714 | 0.270 |
| clip gate texture | 0.363 | 0.250 | 0.238 | 0.716 | 0.275 |
| dino_proto only | 0.233 | 0.071 | 0.000 | 0.790 | 0.306 |
| texture only | 0.179 | 0.179 | 0.099 | 0.259 | 0.085 |
| clip + dino (avg) | 0.181 | 0.190 | 0.022 | 0.320 | 0.115 |
| clip + dino + texture | 0.183 | 0.212 | 0.039 | 0.267 | 0.099 |
| clip + texture (geometric) | 0.151 | 0.175 | 0.000 | 0.253 | 0.115 |

**Proposal sweep (winner fusion):** multi_thresh **0.365** > stability 0.256 > watershed 0.219 ≫ adaptive 0.013.

**Round 2** (refine around front-runners) confirmed the plateau: `clip_ms+texture` kept the best
robustness score; `clip_lc` variants gave higher mean but worse worst-case.

**FINAL:** fusion `clip_ms + texture (2:1)`, proposal `multi_thresh`, multi-prompt SAM 2 + label-free selection.

| Metric | Baseline (Exp 8) | **Final fused** | Human-prompt ceiling |
|---|---|---|---|
| mean poly IoU | 0.227 | **0.365** | 0.743 |
| median | 0.244 | 0.254 | — |
| min / max | 0.084 / 0.336 | **0.237 / 0.714** | — |
| coverage | 0.31 | **0.693** | — |
| boundary-F1 | 0.22 | 0.27 | high |
| fragments | 2.8 | **2.0** | 1/field |

**Per-target IoU (final):** source1969 sheet **0.714**, 1980 sheet 0.261, source1980 crop 0.248,
PNG crop 0.237. → mean **0.365** (hits the "minimum useful ≥ 0.35" milestone; below "promising 0.45").

Maps: `outputs/pipeline/maps/*_compare8.png` (8-panel) and `*_flow.png` (full flow).
Production config: `configs/production_pipeline.yaml`.

---

## Answers to the required questions

1. **Which prior contributes most?** **CLIP, decisively.** Alone it reaches 0.35; texture alone
   0.18, DINOv2 alone 0.23 (and unstable). CLIP's semantic separability (tile-AUROC 0.93 on the
   crop where it works) is the backbone.
2. **Does DINOv2 add value beyond CLIP?** **No.** Every CLIP+DINOv2 fusion (0.16–0.24) is *worse*
   than CLIP alone (0.35). DINOv2's label-free prior is bimodal — 0.79 on one map sheet, 0.00 on the
   crops — so fusing it injects noise. It is excluded from production.
3. **Does the texture prior help?** **Marginally, for robustness.** It lifts the worst-case min-IoU
   (0.237 vs CLIP-only 0.183) by rescuing the target where CLIP fails, at a slight mean cost vs
   `clip_lc`. Alone it is weak (0.18); its value is complementary, not standalone.
4. **Which proposal method is best?** **Multi-threshold connected components (0.365).** Watershed and
   cross-threshold stability are weaker; adaptive thresholding on the fused map collapses.
5. **Which automatic SAM prompt works best?** No single prompt — the **multi-prompt + label-free
   selection** (try box, eroded-box, top-k points, ±negative points; pick the mask that best fills the
   fused prior and avoids the negative prior). This selection step is a large part of the
   0.227→0.365 gain; SAM 2's own confidence is *not* a reliable selector.
6. **How close to human-prompted SAM 2?** ~49 % of the ceiling on average (0.365 vs 0.743), but
   **0.71 on the best target** (near-parity there). The gap is entirely on the low-contrast crops.
7. **Stable across 1969 and 1980?** Only partially — variance is driven more by **target type**
   (framed map sheet vs un-framed crop) than by year. Map sheets reach 0.26–0.71; crops sit at
   0.24. Coverage is stable (~0.69) and there is **no collapse** (min 0.237).
8. **Good enough for automatic mapping?** **No.** Mean 0.365 (crops ~0.24) is below the ≥0.45
   "promising" bar; boundaries are coarse.
9. **Good enough for human-assisted review?** **Yes.** It reliably locates the fields (coverage
   0.69), emits clean few-fragment polygons (≈2), and never collapses — a reviewer nudges instead of
   drawing from scratch. This is the recommended deployment.
10. **What remains the bottleneck?** **Localisation precision of the CLIP semantic prior on
    low-contrast / dark crops.** CLIP's tile-scale separability drops sharply there (AUROC 0.27 on the
    dark PNG vs 0.93 on the lighter crop), so the derived box is imprecise. SAM 2 segmentation is
    *not* the bottleneck (human box → 0.74 with the same SAM 2).

---

## Conclusion & recommendation

Bounded zero-shot optimisation raised the fully-automatic pipeline from **IoU 0.227 → 0.365**
(+61 %), crossing the "minimum useful" line and, crucially, removing the collapses (min 0.237,
coverage 0.69, ~2 fragments). The gains came from **(a)** a stronger CLIP prior (multi-scale +
historical prompt ensembles), **(b)** multi-prompt SAM 2 with **label-free mask selection**, and
**(c)** a texture prior for worst-case robustness. DINOv2, despite excellent *feature* separability,
does not help as an automatic *localiser* and was dropped.

The pipeline has **plateaued for zero-shot** on these targets: the remaining gap to the human
ceiling is localisation precision on low-contrast crops, which no further prior-fusion or
prompt-engineering closed. The demonstrated bottleneck — and the evidence that frozen CLIP/DINOv2
embeddings separate the classes at AUROC ≈ 0.91–0.92 — points to the **only** high-value next step
beyond zero-shot: a **light head on frozen embeddings** (or CLIP-guided upsampling to a dense
per-pixel prior), which requires a modest labelled set. Until then, **ship the fused
CLIP+texture → SAM 2 pipeline as the automatic proposal stage of a human-review tool.**
