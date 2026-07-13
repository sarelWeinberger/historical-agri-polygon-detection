#!/usr/bin/env python3
"""
Experiment 1 — Grounding DINO as an open-vocabulary detector (zero-shot).

Runs every agricultural prompt independently over the ink-free evaluation crops,
sweeps confidence thresholds, records boxes/scores/runtime/VRAM, scores each
against the annotation ground truth, and writes overlays.

Single-GPU discipline: one model, BF16, inference_mode, batch 1, VRAM verified
before load and freed after. No training.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, boxes_to_mask, score_prediction,
                                    gt_boxes_from_mask, overlay, Results, free_gpu)
from src.evaluation import prompts as P

MODEL_ID = "IDEA-Research/grounding-dino-base"
OUT = "outputs/zeroshot/exp1_grounding_dino"
DATA = "outputs/zeroshot/data"


def load_targets():
    return json.load(open(os.path.join(DATA, "eval_targets.json")))


def main():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    os.makedirs(OUT, exist_ok=True)
    targets = load_targets()
    res = Results(os.path.join(OUT, "results.json"))
    # NOTE: Grounding DINO's deformable-attention grid_sample requires fp32 grids;
    # bf16/fp16 casting throws "expected BFloat16 but found Float". The model is
    # small (~700MB), so we run it in fp32 — plenty of headroom on 32GB.
    precision = "fp32 (bf16 unsupported by deformable grid_sample)"

    with gpu_session("grounding-dino-base") as sess:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID).to("cuda").eval()

        for t in targets:
            clean = cv2.imread(t["clean_path"])
            gt_pos = cv2.imread(t["gt_pos_path"], 0)
            gt_neg = cv2.imread(t["gt_neg_path"], 0)
            H, W = gt_pos.shape
            gt_boxes = gt_boxes_from_mask(gt_pos)
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

            for prompt in P.ALL_PROMPTS:
                text = prompt if prompt.endswith(".") else prompt + "."
                inp = proc(images=pil, text=text.lower(), return_tensors="pt").to("cuda")
                t0 = time.time()
                with torch.inference_mode():
                    out = model(**inp)
                dt = time.time() - t0
                post = proc.post_process_grounded_object_detection(
                    out, inp["input_ids"], threshold=0.10, text_threshold=0.10,
                    target_sizes=[(H, W)])[0]
                boxes = post["boxes"].cpu().numpy().tolist()
                scores = post["scores"].cpu().numpy().tolist()

                for thr in P.BOX_THRESHOLDS:
                    keep = [b for b, s in zip(boxes, scores) if s >= thr]
                    ks = [s for s in scores if s >= thr]
                    pmask = boxes_to_mask(keep, H, W)
                    m = score_prediction(pmask, gt_pos, gt_neg,
                                         pred_boxes=keep, gt_boxes=gt_boxes)
                    row = dict(model="grounding-dino-base", exp="exp1", precision=precision,
                               target=t["name"], year=t["year"], prompt=prompt,
                               threshold=thr, n_boxes=len(keep),
                               max_score=round(max(ks), 4) if ks else 0.0,
                               runtime_s=round(dt, 3), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                          for k, v in m.items()})
                    res.add(**row)
                    if thr == P.BOX_THRESHOLDS[1]:  # overlay at the middle threshold
                        title = f"GDINO '{prompt}' thr{thr} n={len(keep)} cov={m['coverage']}"
                        overlay(clean, pmask, gt_pos, gt_neg,
                                os.path.join(OUT, "overlays",
                                             f"{t['name']}__{prompt.replace(' ','_')}.png"),
                                pred_boxes=keep, title=title)
        sess["model_id"] = MODEL_ID
        free_gpu(model)

    # quick summary
    rows = res.rows
    print(f"\n=== Exp1 Grounding DINO: {len(rows)} rows ===")
    # best positive-prompt coverage per target at mid threshold
    for t in targets:
        sub = [r for r in rows if r["target"] == t["name"] and r["threshold"] == 0.25
               and r["prompt"] in P.POSITIVE]
        if not sub: continue
        best = max(sub, key=lambda r: (r["coverage"] or 0))
        print(f"  {t['name']:14s} best pos prompt @0.25: '{best['prompt']}' "
              f"cov={best['coverage']} iou={best['mask_iou']} fp_blk={best['fp_on_black']} "
              f"nbox={best['n_boxes']}")


if __name__ == "__main__":
    main()
