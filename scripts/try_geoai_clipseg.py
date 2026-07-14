#!/usr/bin/env python3
"""
Try geoai's CLIPSegmentation (CLIPSeg, text-prompted DENSE segmentation) zero-shot
on the historical evaluation crops, and score vs GT with our own harness.

CLIPSeg is a dense text->mask model — a candidate fix for the localization
bottleneck our tile-scale CLIP prior hit. No training; GT only for scoring.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "outputs/zeroshot/data"
OUT = "outputs/geoai"
PROMPTS = ["cultivated field", "plowed agricultural field",
           "farmland with cultivation rows", "agricultural field"]


def png_to_gtiff(png, tif):
    import rasterio
    from rasterio.transform import from_origin
    bgr = cv2.imread(png, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    tr = from_origin(0, h, 1, 1)  # identity-ish (pixel coords)
    with rasterio.open(tif, "w", driver="GTiff", height=h, width=w, count=3,
                       dtype="uint8", crs="EPSG:32636", transform=tr) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def score(pred, gp, gn):
    p = pred > 0; pos = gp > 0; neg = gn > 0
    inter = np.logical_and(p, pos).sum()
    iou = inter / (np.logical_or(p, pos).sum() + 1e-6)
    cov = inter / (pos.sum() + 1e-6)
    fpb = np.logical_and(p, neg).sum() / (neg.sum() + 1e-6) if neg.sum() else None
    return dict(iou=round(float(iou), 4), coverage=round(float(cov), 4),
                fp_on_black=(round(float(fpb), 4) if fpb is not None else None),
                pred_area_frac=round(float(p.mean()), 4))


def main():
    import geoai
    print("geoai", getattr(geoai, "__version__", "?"))
    os.makedirs(OUT, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    results = []
    seg = geoai.CLIPSegmentation(tile_size=512, overlap=64)
    for t in targets:
        tif = os.path.join(OUT, f"{t['name']}_clean.tif")
        png_to_gtiff(t["clean_path"], tif)
        gp = cv2.imread(t["gt_pos_path"], 0); gn = cv2.imread(t["gt_neg_path"], 0)
        for prompt in PROMPTS:
            outp = os.path.join(OUT, f"{t['name']}_{prompt.replace(' ','_')}.tif")
            t0 = time.time()
            try:
                seg.segment_image(tif, output_path=outp, text_prompt=prompt,
                                  threshold=0.4, smoothing_sigma=1.0)
                import rasterio
                with rasterio.open(outp) as ds:
                    pred = ds.read(1)
            except Exception as e:
                print(f"  {t['name'][:12]} '{prompt}': ERROR {e}")
                continue
            dt = time.time() - t0
            m = score((pred > 0).astype(np.uint8), gp, gn)
            m.update(target=t["name"], prompt=prompt, runtime_s=round(dt, 2))
            results.append(m)
            print(f"  {t['name'][:12]:12s} '{prompt:28s}' IoU={m['iou']} cov={m['coverage']} "
                  f"fpblk={m['fp_on_black']} area={m['pred_area_frac']} {dt:.1f}s")
    json.dump(results, open(os.path.join(OUT, "clipseg_results.json"), "w"), indent=2, ensure_ascii=False)
    if results:
        import numpy as _np
        best = {}
        for t in targets:
            sub = [r for r in results if r["target"] == t["name"]]
            if sub:
                b = max(sub, key=lambda r: r["iou"]); best[t["name"]] = b["iou"]
        print(f"\nbest-per-target IoU: {best}")
        print(f"mean best IoU: {_np.mean(list(best.values())):.3f}")
        print("fused pipeline reference: mean 0.365, human ceiling 0.743")


if __name__ == "__main__":
    main()
