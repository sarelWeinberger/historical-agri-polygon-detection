#!/usr/bin/env python3
"""
Experiment 5 — OWLv2 (google/owlv2-base-patch16-ensemble), zero-shot open-vocab
detection. Runs the full agricultural prompt set as text queries, sweeps
thresholds, and measures recall / precision / false positives vs the GT.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, boxes_to_mask, score_prediction,
                                    gt_boxes_from_mask, overlay, Results, free_gpu)
from src.evaluation import prompts as P

MODEL_ID = "google/owlv2-base-patch16-ensemble"
OUT = "outputs/zeroshot/exp5_owlv2"
DATA = "outputs/zeroshot/data"
THRESHOLDS = [0.05, 0.10, 0.15]


def main():
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with gpu_session("owlv2-base") as sess:
        proc = Owlv2Processor.from_pretrained(MODEL_ID)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, torch_dtype=dtype).to("cuda").eval()

        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            gt_boxes = gt_boxes_from_mask(gt_pos)
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

            # one query per prompt (run independently)
            for prompt in P.ALL_PROMPTS:
                inp = proc(text=[[prompt]], images=pil, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                t0 = time.time()
                with torch.inference_mode():
                    out = model(**inp)
                dt = time.time() - t0
                post = proc.post_process_grounded_object_detection(
                    out, threshold=0.02, target_sizes=torch.tensor([[H, W]]))[0]
                boxes = post["boxes"].float().cpu().numpy().tolist()
                scores = post["scores"].float().cpu().numpy().tolist()
                for thr in THRESHOLDS:
                    keep = [b for b, s in zip(boxes, scores) if s >= thr]
                    ks = [s for s in scores if s >= thr]
                    pmask = boxes_to_mask(keep, H, W)
                    m = score_prediction(pmask, gt_pos, gt_neg, pred_boxes=keep, gt_boxes=gt_boxes)
                    res.add(model="owlv2-base", exp="exp5", target=t["name"], year=t["year"],
                            prompt=prompt, threshold=thr, n_boxes=len(keep),
                            max_score=round(max(ks), 4) if ks else 0.0, runtime_s=round(dt, 3),
                            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
                    if thr == THRESHOLDS[1] and prompt == "cultivated agricultural field":
                        overlay(clean, pmask, gt_pos, gt_neg,
                                os.path.join(OUT, "overlays", f"{t['name']}__cultivated.png"),
                                pred_boxes=keep,
                                title=f"OWLv2 '{prompt}' thr{thr} n={len(keep)} cov={m['coverage']} iou={m['mask_iou']}")
        free_gpu(model)

    rows = res.rows
    print(f"\n=== Exp5 OWLv2: {len(rows)} rows ===")
    for t in targets:
        sub = [r for r in rows if r["target"] == t["name"] and r["threshold"] == 0.10
               and r["prompt"] in P.POSITIVE]
        if sub:
            best = max(sub, key=lambda r: r["mask_iou"])
            print(f"  {t['name']:14s} best pos@0.10 '{best['prompt']}' iou={best['mask_iou']} "
                  f"cov={best['coverage']} nbox={best['n_boxes']} predArea={best['pred_area_frac']:.2f} "
                  f"fp_blk={best['fp_on_black']}")
    # prompt sensitivity: mean n_boxes for positive vs negative prompts
    pos = [r for r in rows if r["prompt"] in P.POSITIVE and r["threshold"] == 0.10]
    neg = [r for r in rows if r["prompt"] in P.NEGATIVE and r["threshold"] == 0.10]
    print(f"  mean detections @0.10  positive-prompts={np.mean([r['n_boxes'] for r in pos]):.1f}  "
          f"negative-prompts={np.mean([r['n_boxes'] for r in neg]):.1f}")


if __name__ == "__main__":
    main()
