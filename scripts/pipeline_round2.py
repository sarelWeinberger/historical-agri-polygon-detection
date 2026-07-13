#!/usr/bin/env python3
"""Round 2 — refine fusion around the round-1 front-runners (clip_ms/clip_lc + texture).
Dev-set tuning on the 4 existing targets (labelled as such; no generalisation claim)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import gpu_session, free_gpu
from scripts.pipeline_optimize import run_config, summarize, load_priors  # reuse

DATA = "outputs/zeroshot/data"; ROUNDS = "outputs/pipeline/rounds"

REFINED = {
    "clip_ms+tex_2:1":   {"rule": "weighted_avg", "use": ["clip_ms", "texture"], "weights": {"clip_ms": 2, "texture": 1}},
    "clip_lc_only":      {"rule": "weighted_avg", "use": ["clip_lc"]},
    "clip_lc+tex_2:1":   {"rule": "weighted_avg", "use": ["clip_lc", "texture"], "weights": {"clip_lc": 2, "texture": 1}},
    "clip_lc+tex_3:1":   {"rule": "weighted_avg", "use": ["clip_lc", "texture"], "weights": {"clip_lc": 3, "texture": 1}},
    "clip_lc_gate_tex":  {"rule": "gate", "use": ["clip_lc", "texture"]},
    "clipms+cliplc":     {"rule": "weighted_avg", "use": ["clip_ms", "clip_lc"]},
    "clipms+cliplc+tex": {"rule": "weighted_avg", "use": ["clip_ms", "clip_lc", "texture"], "weights": {"clip_ms": 2, "clip_lc": 2, "texture": 1}},
}


def main():
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    prev = json.load(open(os.path.join(ROUNDS, "sweep_results.json")))
    r1_win = prev["WINNER"]["summary"]
    print(f"round-1 winner: clip+texture_avg mean={r1_win['mean_iou']} min={r1_win['min_iou']}")
    out = {}
    with gpu_session("sam2.1-hiera-large") as s:
        from src.evaluation.sam2_helper import SAM2
        sam = SAM2()
        for name, spec in REFINED.items():
            rows = run_config(sam, targets, spec, "multi_thresh")
            summ = summarize(rows)
            score = 0.5 * (summ["mean_iou"] + summ["min_iou"])
            out[name] = {"summary": summ, "spec": spec, "robust_score": round(score, 4),
                         "rows": [{k: v for k, v in r.items() if k != "_pred"} for r in rows],
                         "_rows": rows}
            print(f"  {name:20s} meanIoU={summ['mean_iou']} med={summ['median_iou']} "
                  f"min={summ['min_iou']} max={summ['max_iou']} bF1={summ['mean_bf1']} robust={score:.3f}")
        # best across round1 winner + round2 refinements (by robust score)
        best = max(out, key=lambda n: out[n]["robust_score"])
        r1_score = 0.5 * (r1_win["mean_iou"] + r1_win["min_iou"])
        print(f"\nround-2 best: {best} robust={out[best]['robust_score']} vs round-1 {r1_score:.3f}")
        if out[best]["robust_score"] > r1_score:
            print(f"ADOPT {best} as final winner")
            rows = out[best]["_rows"]
            for r in rows:
                np.save(os.path.join(ROUNDS, f"{r['target']}_best_pred.npy"), r["_pred"])
            final = {"fusion": best, "spec": out[best]["spec"], "method": "multi_thresh",
                     "summary": out[best]["summary"],
                     "rows": [{k: v for k, v in r.items() if k != "_pred"} for r in rows]}
        else:
            print("KEEP round-1 winner (clip+texture_avg)")
            final = {"fusion": "clip+texture_avg",
                     "spec": {"rule": "weighted_avg", "use": ["clip_ms", "texture"], "weights": {"clip_ms": 2, "texture": 1}},
                     "method": "multi_thresh", "summary": r1_win, "rows": prev["WINNER"]["rows"]}
        free_gpu(sam.model)
    out["FINAL"] = final
    json.dump({k: v for k, v in out.items()}, open(os.path.join(ROUNDS, "round2_results.json"), "w"),
              indent=2, ensure_ascii=False, default=lambda o: None)
    print(f"\n### FINAL: {final['fusion']} / {final['method']} -> {final['summary']}")


if __name__ == "__main__":
    main()
