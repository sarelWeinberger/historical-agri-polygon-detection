#!/usr/bin/env python3
"""Aggregate all zero-shot experiment results into one comparison table (md+json).

Columns: model | best prompt/mode | year-scope | recall(coverage) | IoU |
false-positives (fp_on_black / fp_area) | runtime | peak VRAM | note.
Peak VRAM is taken from the recorded per-model figures below (from gpu_session).
"""
import json, os, glob
import numpy as np

ROOT = "outputs/zeroshot"
PEAK_VRAM = {  # MB, observed via gpu_session; precision used
    "grounding-dino-base": (2350, "fp32"),
    "gdino+sam2": (2350, "fp32/bf16"),
    "sam2-prompted": (1424, "bf16"),
    "florence2": (2122, "fp16"),
    "owlv2-base": (440, "bf16"),
    "clip": (1466, "bf16"),
    "siglip": (702, "bf16"),
    "dinov2-base": (986, "bf16"),
}


def load(exp):
    p = os.path.join(ROOT, exp, "results.json")
    return json.load(open(p)) if os.path.exists(p) else []


def summarize():
    out = []

    r1 = load("exp1_grounding_dino")
    pos = [x for x in r1 if x["prompt"] not in ("rocky uncultivated slope", "natural erosion pattern")]
    best = max(pos, key=lambda x: x["mask_iou"]) if pos else {}
    out.append(dict(model="Grounding DINO (base)", mode="open-vocab detect (best prompt/thr)",
                    recall=_mean([x["coverage"] for x in pos if x["coverage"] is not None]),
                    iou=best.get("mask_iou"), fp="whole-scene box; fp_on_black up to 1.0",
                    runtime_s=_mean([x["runtime_s"] for x in r1]),
                    note="prompt-insensitive; boxes cover ~84% of image; pos==neg prompts"))

    r2 = load("exp2_gdino_sam2")
    out.append(dict(model="Grounding DINO + SAM 2", mode="GDINO box -> SAM2 mask",
                    recall=_mean([x["coverage"] for x in r2 if x["coverage"] is not None]),
                    iou=_mean([x["mask_iou"] for x in r2]),
                    fp="fp_on_black ~0.98 (inherits GDINO box)",
                    runtime_s=_mean([x["runtime_s"] for x in r2]),
                    note="SAM2 refines faithfully but GDINO boxes are unusable"))

    r3 = load("exp3_prompted_sam2")
    for v in ["A_single_point", "B_multi_points", "C_pos_neg_points", "D_box"]:
        sub = [x for x in r3 if x["variant"] == v]
        out.append(dict(model=f"Prompted SAM 2 [{v}]", mode="human prompt -> mask",
                        recall=None, iou=_mean([x["field_iou"] for x in sub]),
                        fp=f"fp_on_black {_mean([x['fp_on_black'] for x in sub]):.3f}",
                        runtime_s=_mean([x["runtime_s"] for x in sub]),
                        note="boundary quality given a correct human prompt"))

    r4 = load("exp4_florence2")
    for task in ["<OD>", "<REGION_PROPOSAL>", "phrase_grounding", "referring_seg"]:
        sub = [x for x in r4 if x["task"] == task]
        out.append(dict(model=f"Florence-2 [{task}]", mode="VLM task",
                        recall=_mean([x["coverage"] for x in sub if x["coverage"] is not None]),
                        iou=_mean([x["mask_iou"] for x in sub]),
                        fp="oversized boxes / weak masks",
                        runtime_s=_mean([x["runtime_s"] for x in sub]),
                        note="labels scenes 'airplane/poster/animal' -> out of domain"))

    r5 = load("exp5_owlv2")
    posp = [x for x in r5 if x["threshold"] == 0.05 and x["prompt"] not in
            ("rocky uncultivated slope", "natural erosion pattern")]
    out.append(dict(model="OWLv2 (base)", mode="open-vocab detect",
                    recall=_mean([x["coverage"] for x in posp if x["coverage"] is not None]),
                    iou=max([x["mask_iou"] for x in posp]) if posp else None,
                    fp="terrain prompts fire ~47 boxes; pos prompts ~0",
                    runtime_s=_mean([x["runtime_s"] for x in r5]),
                    note="near-zero recall on cultivation; over-fires on terrain"))

    r6 = load("exp6_clip_siglip")
    for kind in ["clip", "siglip"]:
        sub = [x for x in r6 if x["model"] == kind]
        aucs = [x["tile_auc"] for x in sub if x.get("tile_auc") is not None]
        out.append(dict(model=f"{kind.upper()} tile-scoring", mode="contrastive tiles -> heatmap",
                        recall=_mean([x["coverage"] for x in sub if x["coverage"] is not None]),
                        iou=_mean([x["mask_iou"] for x in sub]),
                        fp=f"tile-AUROC(cult vs hardneg)={_mean(aucs):.3f}",
                        runtime_s=_mean([x["runtime_s"] for x in sub]),
                        note="CLIP separates cultivation from terrain; SigLIP inverted"))

    r7 = load("exp7_dinov2")
    aucs = [x["loo_prototype_auroc"] for x in r7 if x.get("loo_prototype_auroc") is not None]
    out.append(dict(model="DINOv2 (base) embeddings", mode="tile embedding separability",
                    recall=None, iou=None,
                    fp=f"LOO-prototype AUROC={_mean(aucs):.3f} (source1980)",
                    runtime_s=_mean([x["runtime_s"] for x in r7]),
                    note="cultivated vs hard-neg naturally separable (no training)"))

    # exp8 — fully-automatic CLIP/DINOv2 -> region proposal -> SAM2
    r8 = load("exp8_auto")
    best = {"clip": "A_threshold", "dinov2": "D_multithresh"}
    for src, lbl in [("clip", "CLIP → RP → SAM2 (auto)"), ("dinov2", "DINOv2 → RP → SAM2 (auto)")]:
        sub = [x for x in r8 if x["source"] == src and x["method"] == best[src]]
        out.append(dict(model=lbl, mode="automatic prompt (no human, no GT)",
                        recall=_mean([x["coverage"] for x in sub if x["coverage"] is not None]),
                        iou=_mean([x["poly_iou"] for x in sub]),
                        fp=f"boundaryF1={_mean([x['boundary_f1'] for x in sub]):.3f}, "
                           f"frags={_mean([x['fragments'] for x in sub]):.1f}",
                        runtime_s=_mean([x["runtime_s"] for x in sub]),
                        note="localises right fields; ~3x below human prompt (loc. gap)"))

    for row in out:
        m = row["model"].split(" [")[0].split(" tile")[0].split(" (")[0]
        key = {"Grounding DINO": "grounding-dino-base", "Grounding DINO + SAM 2": "gdino+sam2",
               "Prompted SAM 2": "sam2-prompted", "Florence-2": "florence2",
               "OWLv2": "owlv2-base", "CLIP": "clip", "SIGLIP": "siglip",
               "DINOv2": "dinov2-base", "CLIP → RP → SAM2": "clip",
               "DINOv2 → RP → SAM2": "clip"}.get(m)
        pv = PEAK_VRAM.get(key)
        row["peak_vram_mb"], row["precision"] = (pv if pv else (None, None))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def to_md(rows):
    h = "| Model | Mode | Recall (cov) | IoU | False positives / separability | Runtime/img (s) | Peak VRAM (MB) | Note |\n"
    h += "|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        h += (f"| {r['model']} | {r['mode']} | {r['recall']} | {r['iou']} | {r['fp']} | "
              f"{r['runtime_s']} | {r['peak_vram_mb']} ({r['precision']}) | {r['note']} |\n")
    return h


if __name__ == "__main__":
    rows = summarize()
    json.dump(rows, open(os.path.join(ROOT, "comparison_table.json"), "w"), indent=2, ensure_ascii=False)
    md = to_md(rows)
    open(os.path.join(ROOT, "comparison_table.md"), "w").write(md)
    print(md)
