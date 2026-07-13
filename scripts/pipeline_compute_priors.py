#!/usr/bin/env python3
"""
Compute + cache all zero-shot priors, one model at a time.

  CLIP  -> clip_ms (multi-scale ensemble margin) + clip_lc (local contrast)
  DINOv2-> dino_proto (CLIP-agreement prototype similarity, label-free)
           dino_bright (brightness-oriented PC1 baseline, for comparison)
  CPU   -> texture (structure-tensor coherence + Gabor + FFT periodicity)

No GT is used anywhere here. DINOv2 is oriented by agreement with the CLIP prior
(another zero-shot model), never by labels.
"""
from __future__ import annotations

import json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.harness import gpu_session, free_gpu, cuda_free_mb
from src.pipeline import priors as PR

DATA = "outputs/zeroshot/data"
CACHE = "outputs/pipeline/priors"
CLIP_SIZES = [224, 336, 448]
CLIP_OVERLAP = 0.5


def _as_tensor(f):
    if torch.is_tensor(f): return f.float()
    for a in ("text_embeds", "image_embeds", "pooler_output"):
        if getattr(f, a, None) is not None: return getattr(f, a).float()
    return f.last_hidden_state.mean(1).float()


def clip_stage(targets, dtype):
    from transformers import AutoProcessor, AutoModel
    with gpu_session("clip-vit-large") as s:
        proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = AutoModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to("cuda").eval()

        def emb_text(texts):
            inp = proc(text=texts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                f = model.get_text_features(**inp)
            return torch.nn.functional.normalize(_as_tensor(f), dim=-1)

        pos_t, neg_t = emb_text(PR.CLIP_POS), emb_text(PR.CLIP_NEG)
        for t in targets:
            rgb = cv2.cvtColor(cv2.imread(t["clean_path"]), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            scale_maps = []
            for size in CLIP_SIZES:
                coords = PR.tile_grid(H, W, size, CLIP_OVERLAP)
                imgs = [rgb[y:y+size, x:x+size] for (x, y) in coords]
                feats = []
                for i in range(0, len(imgs), 64):
                    inp = proc(images=imgs[i:i+64], return_tensors="pt").to("cuda")
                    inp["pixel_values"] = inp["pixel_values"].to(dtype)
                    with torch.inference_mode():
                        f = model.get_image_features(pixel_values=inp["pixel_values"])
                    feats.append(torch.nn.functional.normalize(_as_tensor(f), dim=-1))
                feats = torch.cat(feats, 0)
                margin = ((feats @ pos_t.T).mean(1) - (feats @ neg_t.T).mean(1)).cpu().numpy()
                scale_maps.append(PR.blend_scale(H, W, coords, margin, size))
            clip_ms = PR.norm01(np.mean(scale_maps, 0))
            np.save(os.path.join(CACHE, f"{t['name']}_clip_ms.npy"), clip_ms)
            np.save(os.path.join(CACHE, f"{t['name']}_clip_lc.npy"), PR.local_contrast(clip_ms))
        free_gpu(model)


def dino_stage(targets, dtype):
    from transformers import AutoImageProcessor, AutoModel
    from sklearn.cluster import KMeans
    TILE, OV = 96, 0.5
    with gpu_session("dinov2-base") as s:
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base", dtype=dtype).to("cuda").eval()
        for t in targets:
            gray = cv2.imread(t["clean_path"], 0)
            rgb = cv2.cvtColor(cv2.imread(t["clean_path"]), cv2.COLOR_BGR2RGB)
            H, W = gray.shape
            coords = PR.tile_grid(H, W, TILE, OV)
            embs, bright = [], []
            for i in range(0, len(coords), 128):
                batch = [cv2.resize(rgb[y:y+TILE, x:x+TILE], (224, 224)) for (x, y) in coords[i:i+128]]
                inp = proc(images=batch, return_tensors="pt").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(dtype)
                with torch.inference_mode():
                    o = model(**inp)
                embs.append(torch.nn.functional.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu().numpy())
            for (x, y) in coords: bright.append(float(gray[y:y+TILE, x:x+TILE].mean()))
            E = np.concatenate(embs, 0); bright = np.array(bright)

            # --- baseline: PC1 oriented dark=high ---
            from sklearn.decomposition import PCA
            pc1 = PCA(1).fit_transform(E - E.mean(0)).ravel()
            if np.corrcoef(pc1, bright)[0, 1] > 0: pc1 = -pc1
            s01 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-6)
            np.save(os.path.join(CACHE, f"{t['name']}_dino_bright.npy"),
                    PR.norm01(PR.blend_scale(H, W, coords, s01, TILE)))

            # --- CLIP-agreement prototype (label-free orientation) ---
            clip_ms = np.load(os.path.join(CACHE, f"{t['name']}_clip_ms.npy"))
            clip_at = np.array([float(clip_ms[min(y+TILE//2, H-1), min(x+TILE//2, W-1)]) for (x, y) in coords])
            k = min(6, max(2, len(E) // 20))
            km = KMeans(k, n_init=5, random_state=0).fit(E)
            # cultivation prototype = cluster centroids weighted by CLIP agreement
            agree = np.array([clip_at[km.labels_ == c].mean() for c in range(k)])
            w = np.exp((agree - agree.max()) / (agree.std() + 1e-6)); w /= w.sum()
            proto = (w[:, None] * km.cluster_centers_).sum(0)
            proto = proto / (np.linalg.norm(proto) + 1e-6)
            sim = E @ proto
            s01 = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
            np.save(os.path.join(CACHE, f"{t['name']}_dino_proto.npy"),
                    PR.norm01(PR.blend_scale(H, W, coords, s01, TILE)))
        free_gpu(model)


def texture_stage(targets):
    for t in targets:
        gray = cv2.imread(t["clean_path"], 0)
        tex = PR.texture_row_prior(gray)
        np.save(os.path.join(CACHE, f"{t['name']}_texture.npy"), tex)


def main():
    os.makedirs(CACHE, exist_ok=True)
    targets = json.load(open(os.path.join(DATA, "eval_targets.json")))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    clip_stage(targets, dtype)
    print(f"[transition] free {cuda_free_mb()[0]:.0f} MB after CLIP")
    dino_stage(targets, dtype)
    print(f"[transition] free {cuda_free_mb()[0]:.0f} MB after DINOv2")
    texture_stage(targets)
    print("texture done (CPU)")
    print("\ncached priors:")
    for t in targets:
        got = [f.split(t["name"]+"_")[1][:-4] for f in os.listdir(CACHE) if f.startswith(t["name"])]
        print(f"  {t['name'][:16]:16s}: {sorted(got)}")


if __name__ == "__main__":
    main()
