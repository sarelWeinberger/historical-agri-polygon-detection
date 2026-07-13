# Zero-Shot Evaluation of Pretrained Vision Models on Historical Aerial Imagery

**Question:** *Can existing pretrained, open-source vision models identify cultivated
agricultural land in 1969/1980 historical aerial imagery **without any training**?*

**Answer (short):** No pretrained model localises fields well *out of the box*. But two
distinct, reusable signals exist zero-shot: **(1) prompted SAM 2 delineates boundaries
excellently once a human/other model says where to look, and (2) CLIP and DINOv2 tile
embeddings separate cultivated land from look-alike terrain at AUROC ≈ 0.91–0.92.** The
missing capability is *localisation/recognition*, not segmentation.

No model was trained, fine-tuned, or adapted. The annotation polygons were used only as
evaluation ground truth (yellow+red = cultivated, black = hard negative).

---

## Setup & compute discipline

- **GPU:** NVIDIA RTX PRO 4500 Blackwell, 32 GB, single card, CUDA 13.2, `sm_120`.
- **Torch:** 2.11.0+cu128 (Blackwell-compatible), `torch.inference_mode()`, **BF16** where
  supported (FP16/FP32 fallbacks noted per model), batch size 1.
- **One model at a time.** Every experiment loads a single model inside a `gpu_session`
  that (a) verifies free VRAM before load, (b) tracks peak, (c) deletes the model +
  `gc.collect()` + `torch.cuda.empty_cache()` + `synchronize()` and re-checks free VRAM
  after. Grounding DINO→SAM 2 (exp 2) is done **sequentially**: GDINO freed (verified
  32.1 GB free) *before* SAM 2 loads. Florence-2 runs in an isolated env (below).
- **Peak VRAM never exceeded 2.4 GB** for any model — the 32 GB budget was never stressed;
  the design would still hold on a much smaller card.
- **Image scope:** the annotated crops + their **ink-free** clean source regions only
  (annotations inpainted out so no model sees a coloured stroke). Full mosaics were *not*
  processed — no zero-shot method was promising enough to justify it (see recommendation).

### Data (4 evaluation targets)
`source1980.JPE`, `דוגמאות…1980.png` (two un-framed crops) and the two ITM map sheets
(`source1969`, `…1908/1980`). Cultivated GT area and, on the two crops, black hard-negative
GT. The map sheets contain only a large yellow field (no black), so hard-negative metrics
come from the two crops.

---

## Comparison table

See `outputs/zeroshot/comparison_table.md`. Key rows:

| Model | Mode | Recall (cov) | IoU | FP / separability | s/img | Peak VRAM |
|---|---|---|---|---|---|---|
| Grounding DINO (base) | open-vocab detect | 1.0 | 0.26 | whole-scene box; fp_on_black→1.0 | 0.15 | 2350 MB fp32 |
| GDINO + SAM 2 | box→mask | 0.97 | 0.20 | fp_on_black ≈0.98 | 0.15 | 2350 MB |
| **Prompted SAM 2 — box** | human prompt→mask | — | **0.74** | fp_on_black 0.09 | 0.15 | 1424 MB bf16 |
| **Prompted SAM 2 — multi-point** | human prompt→mask | — | **0.73** | fp_on_black 0.08 | 0.14 | 1424 MB |
| Prompted SAM 2 — pos+neg pts | human prompt→mask | — | 0.72 | **fp_on_black 0.02** | 0.15 | 1424 MB |
| Prompted SAM 2 — single point | human prompt→mask | — | 0.53 | fp_on_black 0.00 | 0.20 | 1424 MB |
| Florence-2 `<OD>` | VLM detect | 0.63 | 0.16 | labels: airplane/poster/animal | 0.48 | 2122 MB fp16 |
| Florence-2 phrase-grounding | VLM detect | 1.0 | 0.20 | oversized boxes | 0.12 | 2122 MB |
| Florence-2 referring-seg | VLM segment | 0.27 | 0.10 | weak masks | 6.1 | 2122 MB |
| OWLv2 (base) | open-vocab detect | 0.26 | 0.37* | pos prompts ~0 boxes; terrain ~47 | 0.03 | 440 MB bf16 |
| **CLIP-L/14 tile-scoring** | contrastive tiles | 0.68 | 0.28 | **tile-AUROC 0.92** | 0.39 | 1466 MB bf16 |
| SigLIP-base tile-scoring | contrastive tiles | 0.59 | 0.18 | tile-AUROC 0.00 (inverted) | 0.14 | 702 MB bf16 |
| **DINOv2-base embeddings** | embedding separability | — | — | **LOO-AUROC 0.91** | 0.55 | 986 MB bf16 |

\* OWLv2 IoU 0.37 is a single lucky box on one map sheet; recall across positive prompts
is ~0.

---

## Per-model findings

### 1. Grounding DINO (base) — *detects "something is there", cannot localise*
Every prompt — **including the negative controls "rocky uncultivated slope" and "natural
erosion pattern"** — returns essentially one box covering ~84 % of the image (median). The
box trivially "covers" the fields (recall 1.0) but also covers the hard negatives
(fp_on_black up to 1.0) and IoU never exceeds 0.26. **Prompt wording has no effect** (the
best-scoring prompt was often a *negative* one). → **Too many false positives / not usable
for localisation.** BF16 is unsupported (deformable-attention `grid_sample` needs fp32).

### 2. Grounding DINO + SAM 2 — *SAM 2 is fine, the prompt source is not*
SAM 2 faithfully segments whatever box it is given, but GDINO's whole-scene boxes make the
pipeline inherit fp_on_black ≈ 0.98 and IoU ≈ 0.20. **The bottleneck is localisation, not
boundary refinement.**

### 3. Prompted SAM 2 — *useful with human prompts only (the best delineator)*
Given a correct geometric prompt, SAM 2 recovers field boundaries well: box IoU **0.74**
(best single field **0.86**), multi-point **0.73**, single point **0.53**. Adding negative
points on the hard negatives cuts leakage to fp_on_black **0.02**. → **Promising as the
boundary stage of a human-in-the-loop or two-stage tool; it must never decide *where* a
field is.**

### 4. Florence-2 (large) — *out of domain*
`<OD>` labels the scenes **"airplane", "poster", "animal", "human hand"**; dense-region
captioning returns nothing usable. Detection/proposal/phrase-grounding produce oversized
boxes (IoU 0.16–0.20); referring-expression segmentation barely overlaps GT (IoU 0.10,
recall 0.27) and is slow (6 s/img). → **Unsuitable for historical aerial imagery.**
(Runs only under transformers 4.x — isolated in `.venv_flor`.)

### 5. OWLv2 (base) — *wrong-way discrimination (misses cultivation)*
Opposite failure to GDINO: cultivation prompts fire on almost nothing (mean 0.1 boxes at
thr 0.10; 28 % of cases even at 0.05, best IoU 0.37), while the terrain prompts fire **47
boxes each**. → **Near-zero recall on cultivation; over-triggers on terrain. Unsuitable.**

### 6. CLIP / SigLIP tile-scoring — *CLIP is the strongest fully-automatic signal*
**CLIP-L/14**: contrastive margin between the cultivation prompts and terrain prompts
separates cultivated vs hard-negative tiles at **AUROC 0.917**; the heatmap visibly lights
up the real fields and stays cool on the hard negatives. Threshold→polygon IoU is only 0.28
because the naïve Otsu conversion is crude — **the signal is strong, the read-out is what's
weak.** **SigLIP-base fails**: it scores the cultivated tiles as *more* "rocky terrain /
barren slope" than "cultivated land" (AUROC ≈ 0, i.e. inverted) — a genuine model
limitation on grayscale historical imagery, verified with its own logits.
→ **CLIP: promising without training as a region-scoring prior. SigLIP-base: unsuitable.**

### 7. DINOv2 (base) embeddings — *cultivation is naturally separable*
Without any training, leave-one-out nearest-prototype AUROC = **0.912** and 1-NN label
agreement = **0.979** between cultivated and hard-negative tiles (on `source1980`, the one
crop with enough black tiles). PCA/UMAP show cultivated tiles clustering. → **Confirms an
exploitable latent signal**; the other targets lacked hard-negative tiles, so this rests on
one crop and must be re-checked once more negatives exist.

---

## Conclusions per the requested categories

| Model | Verdict |
|---|---|
| Grounding DINO | too many false positives (whole-scene, prompt-insensitive) |
| Grounding DINO + SAM 2 | detects location but not boundaries — actually detects *nothing precisely* |
| **Prompted SAM 2** | **useful with human prompts only** (excellent boundaries) |
| Florence-2 | unsuitable for historical imagery (out of domain) |
| OWLv2 | unsuitable (misses cultivation, over-fires on terrain) |
| **CLIP tile-scoring** | **promising without training** (region prior, AUROC 0.92) |
| SigLIP-base | unsuitable (inverted signal) |
| **DINOv2 embeddings** | **promising without training** (separable latent, AUROC 0.91) |

## Overall recommendation

**No single pretrained model solves the task zero-shot, and open-vocabulary *detectors*
(GDINO, OWLv2, Florence-2) are a dead end on this imagery** — they were trained on natural
photos and either box the whole scene or miss cultivation entirely, with prompt wording
largely irrelevant. Future effort should **not** go into more prompt engineering for those
detectors.

The productive direction is an **ensemble of the two signals that *do* work**, with a human
in the loop:

1. **CLIP (and/or DINOv2) tile-embedding scoring → a cultivation *probability prior*.** Both
   give AUROC ≈ 0.91–0.92 zero-shot. Improve the read-out (finer tiles, smarter
   thresholding, CRF) rather than the model.
2. **Prompted SAM 2 for boundaries**, seeded by peaks of that prior (or by a human click),
   with negative points on rocky/eroded look-alikes to suppress the hard negatives.
3. **Human review** to accept/adjust — realistic given the tiny label set.

**On training:** the embedding separability (CLIP/DINOv2 AUROC ≈ 0.9) is strong evidence
that a *light* classifier on frozen embeddings — or eventual fine-tuning — would work well.
But that requires a real labelled dataset (currently one field/year). So the recommended
sequence is: **(a)** ship the zero-shot CLIP-prior + prompted-SAM 2 + human-review tool now;
**(b)** use it to accelerate collection of many more digitised sites; **(c)** only then
train a frozen-embedding classifier or fine-tune. This matches the earlier data-quality
finding that the current annotation set is far too small for a trustworthy learned model.

### Caveats
- Metrics rest on **4 targets** (2 crops with hard negatives). Directional, not definitive.
- GT polygons were re-vectorised from ~1600 px annotation exports, so absolute IoU ceilings
  are limited by GT coarseness (prompted-SAM 2 masks were often *tighter* than the GT).
- Only base/large checkpoints tested; a larger SigLIP (so400m) or DINOv2-g might differ.
