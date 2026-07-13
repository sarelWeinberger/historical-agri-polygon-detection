#!/usr/bin/env python3
"""
Experiment 4 — Florence-2 (microsoft/Florence-2-large), zero-shot.

Probes four capabilities:
  * <OD>                              generic object detection
  * <REGION_PROPOSAL>                 class-agnostic region proposals
  * <DENSE_REGION_CAPTION>            what does it *call* each region?
  * <CAPTION_TO_PHRASE_GROUNDING>     open-vocab detection from agri phrases
  * <REFERRING_EXPRESSION_SEGMENTATION> polygon mask from an agri expression

Detection/proposal boxes and referring-seg polygons are scored vs the GT.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import (gpu_session, boxes_to_mask, polys_to_mask,
                                    score_prediction, gt_boxes_from_mask, overlay,
                                    Results, free_gpu)
from src.evaluation import prompts as P

MODEL_ID = "microsoft/Florence-2-large"
OUT = "outputs/zeroshot/exp4_florence2"
DATA = "outputs/zeroshot/data"
GROUND_PHRASES = ["cultivated agricultural field", "plowed field",
                  "field with cultivation rows", "terraced field"]


def run_task(model, proc, pil, task, text="", dtype=torch.float16):
    prompt = task + text
    inp = proc(text=prompt, images=pil, return_tensors="pt").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    with torch.inference_mode():
        ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                             max_new_tokens=1024, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return proc.post_process_generation(txt, task=task, image_size=(pil.width, pil.height))


def _florence_compat_shim():
    """Florence-2 remote code predates transformers 5.x and reads a few config
    attributes before the base __init__ sets them. Provide safe class-level
    defaults so its config can construct under the newer transformers."""
    from transformers import PretrainedConfig
    for attr, default in [("forced_bos_token_id", None), ("forced_eos_token_id", None),
                          ("force_bos_token_to_be_generated", False)]:
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, default)


def main():
    from transformers import AutoProcessor, AutoModelForCausalLM
    _florence_compat_shim()
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))
    dtype = torch.float16  # Florence-2 remote code is validated in fp16

    with gpu_session("Florence-2-large") as sess:
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=dtype).to("cuda").eval()

        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            gt_boxes = gt_boxes_from_mask(gt_pos)
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

            # --- box-producing tasks ---
            for task in ["<OD>", "<REGION_PROPOSAL>", "<DENSE_REGION_CAPTION>"]:
                t0 = time.time(); out = run_task(model, proc, pil, task, dtype=dtype); dt = time.time()-t0
                d = out.get(task, {})
                boxes = d.get("bboxes", []); labels = d.get("labels", [])
                pmask = boxes_to_mask(boxes, H, W)
                m = score_prediction(pmask, gt_pos, gt_neg, pred_boxes=boxes, gt_boxes=gt_boxes)
                res.add(model="florence2", exp="exp4", target=t["name"], year=t["year"],
                        task=task, prompt=task, n_boxes=len(boxes),
                        labels=labels[:12], runtime_s=round(dt, 3),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
                if task == "<OD>":
                    overlay(clean, pmask, gt_pos, gt_neg,
                            os.path.join(OUT, "overlays", f"{t['name']}_OD.png"),
                            pred_boxes=boxes, title=f"Florence2 <OD> n={len(boxes)} cov={m['coverage']}")

            # --- phrase grounding (open-vocab detection) ---
            for phrase in GROUND_PHRASES:
                t0 = time.time()
                out = run_task(model, proc, pil, "<CAPTION_TO_PHRASE_GROUNDING>", phrase, dtype)
                dt = time.time()-t0
                d = out.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
                boxes = d.get("bboxes", [])
                pmask = boxes_to_mask(boxes, H, W)
                m = score_prediction(pmask, gt_pos, gt_neg, pred_boxes=boxes, gt_boxes=gt_boxes)
                res.add(model="florence2", exp="exp4", target=t["name"], year=t["year"],
                        task="phrase_grounding", prompt=phrase, n_boxes=len(boxes),
                        runtime_s=round(dt, 3),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

            # --- referring expression segmentation (polygons) ---
            for phrase in GROUND_PHRASES:
                t0 = time.time()
                out = run_task(model, proc, pil, "<REFERRING_EXPRESSION_SEGMENTATION>", phrase, dtype)
                dt = time.time()-t0
                d = out.get("<REFERRING_EXPRESSION_SEGMENTATION>", {})
                polys = d.get("polygons", [])
                flat = []
                for inst in polys:
                    for ring in inst:
                        pts = np.array(ring).reshape(-1, 2)
                        if len(pts) >= 3: flat.append(pts.tolist())
                pmask = polys_to_mask(flat, H, W)
                m = score_prediction(pmask, gt_pos, gt_neg)
                res.add(model="florence2", exp="exp4", target=t["name"], year=t["year"],
                        task="referring_seg", prompt=phrase, n_polys=len(flat),
                        runtime_s=round(dt, 3),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
                if phrase == GROUND_PHRASES[0]:
                    overlay(clean, pmask, gt_pos, gt_neg,
                            os.path.join(OUT, "overlays", f"{t['name']}_refseg.png"),
                            title=f"Florence2 refseg '{phrase}' iou={m['mask_iou']} cov={m['coverage']}")
        free_gpu(model)

    rows = res.rows
    print(f"\n=== Exp4 Florence-2: {len(rows)} rows ===")
    for task in ["<OD>", "<REGION_PROPOSAL>", "phrase_grounding", "referring_seg"]:
        sub = [r for r in rows if r["task"] == task]
        if sub:
            cov = np.mean([r["coverage"] for r in sub if r["coverage"] is not None])
            iou = np.mean([r["mask_iou"] for r in sub])
            nb = np.mean([r.get("n_boxes", r.get("n_polys", 0)) for r in sub])
            print(f"  {task:18s} mean cov={cov:.3f} iou={iou:.3f} mean_dets={nb:.1f} (n={len(sub)})")
    # sample dense-caption labels to see what it calls the terrain
    dc = [r for r in rows if r["task"] == "<DENSE_REGION_CAPTION>"]
    if dc:
        print("  dense-caption sample labels:", dc[0].get("labels"))


if __name__ == "__main__":
    main()
