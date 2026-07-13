| Model | Mode | Recall (cov) | IoU | False positives / separability | Runtime/img (s) | Peak VRAM (MB) | Note |
|---|---|---|---|---|---|---|---|
| Grounding DINO (base) | open-vocab detect (best prompt/thr) | 1.0 | 0.2634 | whole-scene box; fp_on_black up to 1.0 | 0.1491 | 2350 (fp32) | prompt-insensitive; boxes cover ~84% of image; pos==neg prompts |
| Grounding DINO + SAM 2 | GDINO box -> SAM2 mask | 0.9691 | 0.1955 | fp_on_black ~0.98 (inherits GDINO box) | 0.1496 | 2350 (fp32/bf16) | SAM2 refines faithfully but GDINO boxes are unusable |
| Prompted SAM 2 [A_single_point] | human prompt -> mask | None | 0.5255 | fp_on_black 0.000 | 0.1964 | 1424 (bf16) | boundary quality given a correct human prompt |
| Prompted SAM 2 [B_multi_points] | human prompt -> mask | None | 0.7254 | fp_on_black 0.084 | 0.1424 | 1424 (bf16) | boundary quality given a correct human prompt |
| Prompted SAM 2 [C_pos_neg_points] | human prompt -> mask | None | 0.7207 | fp_on_black 0.023 | 0.1482 | 1424 (bf16) | boundary quality given a correct human prompt |
| Prompted SAM 2 [D_box] | human prompt -> mask | None | 0.7433 | fp_on_black 0.090 | 0.1469 | 1424 (bf16) | boundary quality given a correct human prompt |
| Florence-2 [<OD>] | VLM task | 0.6321 | 0.1562 | oversized boxes / weak masks | 0.4843 | 2122 (fp16) | labels scenes 'airplane/poster/animal' -> out of domain |
| Florence-2 [<REGION_PROPOSAL>] | VLM task | 1.0 | 0.1697 | oversized boxes / weak masks | 0.0978 | 2122 (fp16) | labels scenes 'airplane/poster/animal' -> out of domain |
| Florence-2 [phrase_grounding] | VLM task | 1.0 | 0.1988 | oversized boxes / weak masks | 0.1176 | 2122 (fp16) | labels scenes 'airplane/poster/animal' -> out of domain |
| Florence-2 [referring_seg] | VLM task | 0.2732 | 0.1014 | oversized boxes / weak masks | 6.1463 | 2122 (fp16) | labels scenes 'airplane/poster/animal' -> out of domain |
| OWLv2 (base) | open-vocab detect | 0.264 | 0.3656 | terrain prompts fire ~47 boxes; pos prompts ~0 | 0.0262 | 440 (bf16) | near-zero recall on cultivation; over-fires on terrain |
| CLIP tile-scoring | contrastive tiles -> heatmap | 0.6795 | 0.2798 | tile-AUROC(cult vs hardneg)=0.917 | 0.392 | 1466 (bf16) | CLIP separates cultivation from terrain; SigLIP inverted |
| SIGLIP tile-scoring | contrastive tiles -> heatmap | 0.5923 | 0.1814 | tile-AUROC(cult vs hardneg)=0.000 | 0.14 | 702 (bf16) | CLIP separates cultivation from terrain; SigLIP inverted |
| DINOv2 (base) embeddings | tile embedding separability | None | None | LOO-prototype AUROC=0.912 (source1980) | 0.5505 | 986 (bf16) | cultivated vs hard-neg naturally separable (no training) |
| CLIP → RP → SAM2 (auto) | automatic prompt (no human, no GT) | 0.3047 | 0.2268 | boundaryF1=0.219, frags=2.8 | 0.939 | 1466 (bf16) | localises right fields; ~3x below human prompt (loc. gap) |
| DINOv2 → RP → SAM2 (auto) | automatic prompt (no human, no GT) | 0.3466 | 0.2098 | boundaryF1=0.239, frags=2.8 | 0.8685 | 1466 (bf16) | localises right fields; ~3x below human prompt (loc. gap) |
