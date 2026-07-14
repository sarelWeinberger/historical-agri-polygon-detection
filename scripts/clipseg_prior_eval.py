#!/usr/bin/env python3
"""Diagnose CLIPSeg as a dense prior: raw probability (prompt ensemble), separability
AUROC vs GT, and best-threshold IoU. Determines whether CLIPSeg's dense output could
help even if its default threshold is miscalibrated on out-of-domain grayscale."""
import json, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "outputs/zeroshot/data"; OUT = "outputs/geoai"
POS = ["cultivated field", "plowed agricultural field", "farmland with cultivation rows",
       "agricultural terraces", "worked farmland field"]


def auroc(s, y):
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    return float((r[y == 1].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def main():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda").eval()
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for t in targets:
        rgb = cv2.cvtColor(cv2.imread(t["clean_path"]), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        gp = cv2.imread(t["gt_pos_path"], 0); gn = cv2.imread(t["gt_neg_path"], 0)
        from PIL import Image
        pil = Image.fromarray(rgb)
        probs = []
        for p in POS:
            inp = proc(text=[p], images=[pil], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = model(**inp).logits
            pr = torch.sigmoid(logits).float().cpu().numpy()
            pr = cv2.resize(pr.squeeze(), (W, H))
            probs.append(pr)
        heat = np.max(probs, 0)                       # ensemble = max over prompts
        np.save(os.path.join(OUT, f"{t['name']}_clipseg_prob.npy"), heat)
        # separability
        yl = np.full(heat.shape, -1, np.int8); yl[gp > 0] = 1; yl[gn > 0] = 0
        idx = np.where(yl >= 0)
        auc_hn = None
        if (yl[idx] == 0).sum() > 0:
            s = heat[idx]; y = yl[idx]
            sub = np.random.RandomState(0).choice(len(s), min(len(s), 20000), replace=False)
            auc_hn = auroc(s[sub], y[sub])
        ybg = (gp > 0).astype(np.int8).ravel()
        sub = np.random.RandomState(0).choice(len(ybg), 20000, replace=False)
        auc_bg = auroc(heat.ravel()[sub], ybg[sub])
        # best-threshold IoU
        best_iou, best_thr = 0, 0
        for thr in np.linspace(0.1, 0.7, 13):
            pred = heat >= thr
            iou = np.logical_and(pred, gp > 0).sum() / (np.logical_or(pred, gp > 0).sum() + 1e-6)
            if iou > best_iou: best_iou, best_thr = iou, thr
        rows.append(dict(target=t["name"], auc_vs_hardneg=auc_hn,
                         auc_vs_background=round(auc_bg, 3) if auc_bg else None,
                         best_iou=round(float(best_iou), 3), best_thr=round(float(best_thr), 2),
                         heat_range=[round(float(heat.min()), 3), round(float(heat.max()), 3)]))
        print(f"  {t['name'][:14]:14s} AUC(vs hardneg)={auc_hn} AUC(vs bg)={rows[-1]['auc_vs_background']} "
              f"bestIoU={rows[-1]['best_iou']}@thr{rows[-1]['best_thr']} range={rows[-1]['heat_range']}")
    json.dump(rows, open(os.path.join(OUT, "clipseg_prior_eval.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nmean best-threshold IoU (oracle threshold): {np.mean([r['best_iou'] for r in rows]):.3f}")
    print(f"mean AUC vs background: {np.mean([r['auc_vs_background'] for r in rows if r['auc_vs_background']]):.3f}")
    print("reference: our CLIP tile prior AUROC(vs hardneg) 0.93 (src1980); fused pipeline IoU 0.365")


if __name__ == "__main__":
    main()
