#!/usr/bin/env python3
"""
Experiment 7 — DINOv2 embedding separability (no training).

Extract DINOv2 CLS embeddings for overlapping tiles, label each tile by the GT
(cultivated / hard-negative / other), and ask: do cultivated tiles *naturally*
separate from rocky/erosion/other terrain in embedding space?

Quantified WITHOUT training via leave-one-out nearest-prototype AUROC and 1-NN
label agreement. PCA + UMAP used only for visualization.
"""
from __future__ import annotations

import json, os, sys, time
import cv2, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import gpu_session, Results, free_gpu

MODEL_ID = "facebook/dinov2-base"
OUT = "outputs/zeroshot/exp7_dinov2"
DATA = "outputs/zeroshot/data"
TILE, STRIDE = 112, 56   # dense tiling; each tile resized to 224 for DINOv2


def tiles_and_labels(rgb, gt_pos, gt_neg):
    H, W = gt_pos.shape
    coords, labs = [], []
    for y in range(0, H - TILE + 1, STRIDE):
        for x in range(0, W - TILE + 1, STRIDE):
            cy, cx = y + TILE // 2, x + TILE // 2
            frac_pos = (gt_pos[y:y+TILE, x:x+TILE] > 0).mean()
            frac_neg = (gt_neg[y:y+TILE, x:x+TILE] > 0).mean()
            if frac_pos > 0.5:
                lab = 1
            elif frac_neg > 0.5:
                lab = 0
            else:
                lab = -1
            coords.append((x, y)); labs.append(lab)
    return coords, np.array(labs)


def loo_prototype_auroc(emb, labs):
    """Leave-one-out: score each pos/neg tile by (sim to other-pos mean - sim to
    other-neg mean); AUROC of that score vs true label. No training."""
    idx = np.where(labs >= 0)[0]
    if len(np.unique(labs[idx])) < 2:
        return None
    E = emb[idx]; y = labs[idx]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    scores = []
    for i in range(len(idx)):
        mask = np.ones(len(idx), bool); mask[i] = False
        pm = E[mask & (y == 1)].mean(0); nm = E[mask & (y == 0)].mean(0)
        scores.append(E[i] @ pm - E[i] @ nm)
    scores = np.array(scores)
    pos = scores[y == 1]; neg = scores[y == 0]
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
    auc = (ranks[y == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    # 1-NN LOO agreement
    S = E @ E.T; np.fill_diagonal(S, -1)
    nn = S.argmax(1); acc = (y[nn] == y).mean()
    return float(auc), float(acc)


def main():
    from transformers import AutoImageProcessor, AutoModel
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    res = Results(os.path.join(OUT, "results.json"))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with gpu_session("dinov2-base") as sess:
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype).to("cuda").eval()

        for t in targets:
            clean = cv2.imread(t["clean_path"]); rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            gt_pos = cv2.imread(t["gt_pos_path"], 0); gt_neg = cv2.imread(t["gt_neg_path"], 0)
            coords, labs = tiles_and_labels(rgb, gt_pos, gt_neg)
            t0 = time.time()
            embs = []
            for i in range(0, len(coords), 128):
                batch = [cv2.resize(rgb[y:y+TILE, x:x+TILE], (224, 224)) for (x, y) in coords[i:i+128]]
                inp = proc(images=batch, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                with torch.inference_mode():
                    out = model(**inp)
                cls = out.last_hidden_state[:, 0].float().cpu().numpy()
                embs.append(cls)
            emb = np.concatenate(embs, 0)
            dt = time.time() - t0

            sep = loo_prototype_auroc(emb, labs)
            npos = int((labs == 1).sum()); nneg = int((labs == 0).sum())
            row = dict(model="dinov2-base", exp="exp7", target=t["name"], year=t["year"],
                       n_tiles=len(coords), n_pos_tiles=npos, n_neg_tiles=nneg,
                       runtime_s=round(dt, 3))
            if sep:
                row["loo_prototype_auroc"] = round(sep[0], 4)
                row["loo_1nn_agreement"] = round(sep[1], 4)
            res.add(**row)

            # PCA + UMAP viz coloured by label
            try:
                from sklearn.decomposition import PCA
                p = PCA(n_components=min(50, emb.shape[0]-1)).fit_transform(
                    (emb - emb.mean(0)) / (emb.std(0) + 1e-6))
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                col = np.array(["#888888", "#d62728", "#2ca02c"])  # other, neg(red), pos(green)
                cidx = np.where(labs == -1, 0, np.where(labs == 0, 1, 2))
                ax[0].scatter(p[:, 0], p[:, 1], c=col[cidx], s=8)
                ax[0].set_title(f"{t['name']} PCA (green=cultivated, red=hard-neg)")
                try:
                    import umap
                    u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(p)
                    ax[1].scatter(u[:, 0], u[:, 1], c=col[cidx], s=8); ax[1].set_title("UMAP")
                except Exception as e:
                    ax[1].text(0.1, 0.5, f"UMAP n/a: {e}")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT, "plots", f"{t['name']}_embed.png"), dpi=90)
                plt.close()
            except Exception as e:
                print("viz failed", e)
        free_gpu(model)

    rows = res.rows
    print(f"\n=== Exp7 DINOv2 embeddings: {len(rows)} rows ===")
    for r in rows:
        print(f"  {r['target']:14s} tiles={r['n_tiles']} pos={r['n_pos_tiles']} neg={r['n_neg_tiles']} "
              f"LOO-AUROC={r.get('loo_prototype_auroc')} 1NN-agree={r.get('loo_1nn_agreement')}")


if __name__ == "__main__":
    main()
