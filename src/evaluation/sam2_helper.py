"""Thin SAM2 wrapper (transformers Sam2Model) for the zero-shot experiments.

SAM2 only refines boundaries from geometric prompts (points / boxes). It never
decides whether a region is cultivated — that decision comes from the prompt
source (Grounding DINO boxes in exp2, human clicks in exp3).
"""
from __future__ import annotations

import numpy as np
import torch

SAM2_ID = "facebook/sam2.1-hiera-large"


class SAM2:
    def __init__(self, model_id: str = SAM2_ID):
        from transformers import Sam2Processor, Sam2Model
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to("cuda").eval()
        self.model_id = model_id

    @torch.inference_mode()
    def segment_boxes(self, rgb, boxes):
        """boxes: list[[x0,y0,x1,y1]] -> (union_mask uint8, per-box iou scores)."""
        if not boxes:
            return np.zeros(rgb.shape[:2], np.uint8), []
        inp = self.proc(images=rgb, input_boxes=[[list(map(float, b)) for b in boxes]],
                        return_tensors="pt").to("cuda")
        out = self.model(**inp, multimask_output=False)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        scores = out.iou_scores.flatten().tolist()
        union = np.zeros(rgb.shape[:2], np.uint8)
        for i in range(masks.shape[0]):
            union |= (masks[i, 0].numpy() > 0).astype(np.uint8)
        return union, scores

    @torch.inference_mode()
    def segment_points(self, rgb, points, labels):
        """points: [[x,y],...]; labels: [1 pos / 0 neg] -> (mask uint8, score)."""
        inp = self.proc(images=rgb, input_points=[[[list(map(float, p)) for p in points]]],
                        input_labels=[[list(map(int, labels))]],
                        return_tensors="pt").to("cuda")
        out = self.model(**inp, multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        scores = out.iou_scores.flatten().tolist()
        # multimask: pick highest-scoring mask
        best = int(np.argmax(scores))
        return (masks[0, best].numpy() > 0).astype(np.uint8), float(scores[best])
