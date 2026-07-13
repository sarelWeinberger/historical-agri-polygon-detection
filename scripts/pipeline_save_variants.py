#!/usr/bin/env python3
"""Save per-target predictions for the named pipeline variants needed by the maps."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import gpu_session, free_gpu
from scripts.pipeline_optimize import run_config

DATA = "outputs/zeroshot/data"; ROUNDS = "outputs/pipeline/rounds"
VARIANTS = {
    "clip_only":  {"rule": "weighted_avg", "use": ["clip_ms"]},
    "dino_only":  {"rule": "weighted_avg", "use": ["dino_proto"]},
    "clipdino":   {"rule": "weighted_avg", "use": ["clip_ms", "dino_proto"]},
    "fulltex":    {"rule": "weighted_avg", "use": ["clip_ms", "dino_proto", "texture"], "weights": {"clip_ms": 2, "dino_proto": 1, "texture": 1}},
}


def main():
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    meta = {}
    with gpu_session("sam2.1-hiera-large") as s:
        from src.evaluation.sam2_helper import SAM2
        sam = SAM2()
        for vname, spec in VARIANTS.items():
            rows = run_config(sam, targets, spec, "multi_thresh")
            for r in rows:
                np.save(os.path.join(ROUNDS, f"{r['target']}_{vname}_pred.npy"), r["_pred"])
                meta.setdefault(r["target"], {})[vname] = r["poly_iou"]
            print(f"  {vname}: " + ", ".join(f"{r['target'][:10]}={r['poly_iou']}" for r in rows))
        free_gpu(sam.model)
    json.dump(meta, open(os.path.join(ROUNDS, "variant_ious.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
