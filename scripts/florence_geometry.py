"""Florence-2 polygon geometry (run inside the isolated transformers-4.x venv)."""
import os
import cv2, numpy as np, torch
from PIL import Image

from src.evaluation.harness import gpu_session, boxes_to_mask, polys_to_mask, free_gpu

MODEL_ID = "microsoft/Florence-2-large"
GROUND = ["cultivated agricultural field", "plowed field", "field with cultivation rows"]


def _shim():
    from transformers import PretrainedConfig
    for a, d in [("forced_bos_token_id", None), ("forced_eos_token_id", None),
                 ("force_bos_token_to_be_generated", False)]:
        if not hasattr(PretrainedConfig, a):
            setattr(PretrainedConfig, a, d)


def _task(model, proc, pil, task, text="", dtype=torch.float16):
    inp = proc(text=task + text, images=pil, return_tensors="pt").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    with torch.inference_mode():
        ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                             max_new_tokens=1024, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return proc.post_process_generation(txt, task=task, image_size=(pil.width, pil.height))


def run_florence(targets, geom, add):
    from transformers import AutoProcessor, AutoModelForCausalLM
    _shim()
    dtype = torch.float16
    with gpu_session("Florence-2-large") as s:
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=dtype).to("cuda").eval()
        for t in targets:
            clean = cv2.imread(t["clean_path"]); H, W = clean.shape[:2]
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            pil = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            # phrase grounding -> union of boxes
            boxes = []
            for ph in GROUND:
                out = _task(model, proc, pil, "<CAPTION_TO_PHRASE_GROUNDING>", ph, dtype)
                boxes += out.get("<CAPTION_TO_PHRASE_GROUNDING>", {}).get("bboxes", [])
            add(geom, t["name"], "florence2_grounding", boxes_to_mask(boxes, H, W), gt_pos, gt_neg)
            # referring-expression segmentation -> polygons
            flat = []
            for ph in GROUND:
                out = _task(model, proc, pil, "<REFERRING_EXPRESSION_SEGMENTATION>", ph, dtype)
                for inst in out.get("<REFERRING_EXPRESSION_SEGMENTATION>", {}).get("polygons", []):
                    for ring in inst:
                        pts = np.array(ring).reshape(-1, 2)
                        if len(pts) >= 3: flat.append(pts.tolist())
            add(geom, t["name"], "florence2_refseg", polys_to_mask(flat, H, W), gt_pos, gt_neg)
        free_gpu(model)
