"""
Milestone 1 — self-contained HTML annotation viewer / manual validation UI.

For every annotated image it shows four panels side by side:
    Source (ink)  |  Extracted class mask  |  Cultivated binary mask  |  Overlay
plus the per-image metadata, polygon table and review flags.

The page is portable: it copies down-scaled copies of the annotated sources and
references the pipeline's own mask/overlay PNGs with URL-encoded relative paths,
so the whole `outputs/milestone1/` folder can be zipped and opened anywhere.
"""
from __future__ import annotations

import html
import json
import os
import shutil
from urllib.parse import quote

import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _copy_scaled(src_path, dst_path, long=1400):
    im = Image.open(src_path)
    im.thumbnail((long, long))
    im.convert("RGB").save(dst_path)


def _rel(base, path):
    return quote(os.path.relpath(path, base))


def build_viewer(out_dir: str):
    summary = json.load(open(os.path.join(out_dir, "dataset_summary.json")))
    manifest = {im["filename"]: im for im in
                json.load(open(os.path.join(out_dir, "manifest.json")))["images"]}
    assets = os.path.join(out_dir, "viewer_sources")
    os.makedirs(assets, exist_ok=True)

    cards = []
    for im in summary["images"]:
        if im["role"] != "annotated":
            continue
        fn = im["filename"]
        stem = "".join(c if c.isalnum() or c in "._-" else "_"
                       for c in os.path.splitext(fn)[0])
        src_scaled = os.path.join(assets, stem + "_src.png")
        _copy_scaled(manifest[fn]["path"], src_scaled)

        poly = json.load(open(os.path.join(out_dir, "polygons", stem + ".json")))
        rows = ""
        for p in poly["polygons"]:
            flag = ' <span class="rev">REVIEW</span>' if p["review_required"] else ""
            rows += (f"<tr><td class='c-{p['color']}'>{p['color']}</td>"
                     f"<td>{p['class']}</td><td>{p['area_px']:.0f}</td>"
                     f"<td>{p['compactness']}</td><td>{p['is_closed_loop']}</td>"
                     f"<td>{len(p['points'])}</td><td>{flag}</td></tr>")

        geo = im.get("georef")
        geo_html = ""
        if geo:
            geo_html = (f"<div class='geo'>ITM/EPSG:2039 &middot; RMSE "
                        f"{geo['rmse_m']} m &middot; {geo['resolution_m_per_px']} m/px "
                        f"&middot; {geo['n_gcps']} GCPs "
                        f"<span class='prov'>provisional</span></div>")

        def img(path, cap):
            if not path or not os.path.exists(path):
                return f"<figure><div class='missing'>n/a</div><figcaption>{cap}</figcaption></figure>"
            return (f"<figure><a href='{_rel(out_dir, path)}' target='_blank'>"
                    f"<img src='{_rel(out_dir, path)}'></a>"
                    f"<figcaption>{cap}</figcaption></figure>")

        masks = im.get("masks", {})
        panels = "".join([
            img(src_scaled, "Source (annotation ink)"),
            img(masks.get("class_mask_preview"), "Extracted class mask"),
            img(masks.get("cultivated"), "Cultivated mask (binary)"),
            img(im.get("overlay"), "Extraction overlay"),
        ])
        cc = im.get("per_color_counts", {})
        cards.append(f"""
        <section class="card">
          <h2>{html.escape(fn)}</h2>
          <div class="meta">
            <span class="tag">{im['role']}/{im['subtype']}</span>
            <span class="tag">year {im.get('year')}</span>
            <span class="tag">yellow {cc.get('yellow',0)}</span>
            <span class="tag">red {cc.get('red',0)}</span>
            <span class="tag">black {cc.get('black',0)}</span>
          </div>
          {geo_html}
          <div class="panels">{panels}</div>
          <table><thead><tr><th>colour</th><th>class</th><th>area px</th>
            <th>compact</th><th>closed</th><th>verts</th><th>flag</th></tr></thead>
            <tbody>{rows}</tbody></table>
          <details><summary>notes</summary><ul>{''.join(f'<li>{html.escape(n)}</li>' for n in im.get('notes',[]))}</ul></details>
        </section>""")

    tot = summary["totals"]
    page = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Milestone 1 — Annotation Viewer</title>
    <style>
      body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#111;color:#eee}}
      header{{padding:18px 24px;background:#1b1b1b;border-bottom:1px solid #333}}
      h1{{margin:0 0 6px}} .sub{{color:#9ab}}
      .card{{margin:20px 24px;padding:16px;background:#181818;border:1px solid #2a2a2a;border-radius:8px}}
      h2{{margin:0 0 8px;font-size:16px;word-break:break-all}}
      .meta .tag,.prov,.rev{{display:inline-block;padding:2px 8px;margin:2px;border-radius:10px;font-size:12px}}
      .tag{{background:#243;color:#adf}} .prov{{background:#530;color:#fda}}
      .rev{{background:#600;color:#fbb}}
      .geo{{color:#8cf;font-size:13px;margin:6px 0}}
      .panels{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}}
      figure{{margin:0}} img{{width:100%;border:1px solid #333;background:#000}}
      figcaption{{font-size:12px;color:#9ab;text-align:center;padding:4px}}
      .missing{{height:120px;display:flex;align-items:center;justify-content:center;color:#666;border:1px dashed #444}}
      table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
      th,td{{border:1px solid #2c2c2c;padding:4px 8px;text-align:left}}
      .c-yellow{{color:#ee0}} .c-red{{color:#f66}} .c-black{{color:#9cf}}
      @media(max-width:900px){{.panels{{grid-template-columns:repeat(2,1fr)}}}}
    </style></head><body>
    <header><h1>Milestone 1 — Annotation Extraction Viewer</h1>
      <div class="sub">Agricultural polygon detection from historical aerial imagery &middot;
      totals: yellow {tot['yellow']} / red {tot['red']} / black {tot['black']}
      &middot; cultivated polygons {tot['cultivated_polys']} &middot; hard-negatives {tot['hardneg_polys']}</div>
      <div class="sub">Legend: yellow &amp; red &rarr; cultivated_area &middot; black &rarr; hard_negative (always human-review)</div>
    </header>
    {''.join(cards)}
    </body></html>"""
    out = os.path.join(out_dir, "viewer.html")
    with open(out, "w") as f:
        f.write(page)
    return out


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/milestone1"
    print("viewer ->", build_viewer(d))
