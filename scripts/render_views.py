"""Render QA views: satellite basemap with the CropScan and eSurvey layers on top.

Each view is two panels side by side: the plain satellite image (left) and the
same image with the layers drawn on it (right). When some layer names start
with "out", the input layers go on the left and the "out" layers (filled) on
the right.

Usage:
    python scripts/render_views.py --views views.csv --out qa/views \
        --layer cropscan=data/cache/cropscan_utm.parquet:yellow \
        --layer esurvey=data/cache/esurvey_utm.parquet:red

views.csv columns: name, lat, lon, half_m (half width of the view in metres).
"""
import argparse
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box

TILE_URL = "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
R_EARTH = 6378137.0
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))


def tile(x, y, z):
    r = SESSION.get(TILE_URL.format(s=(x + y) % 4, x=x, y=y, z=z), timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def merc_to_tile(mx, my, z):
    n = 2 ** z
    return (mx + math.pi * R_EARTH) / (2 * math.pi * R_EARTH) * n, (math.pi * R_EARTH - my) / (2 * math.pi * R_EARTH) * n


def basemap(bounds, z, pool):
    """Mosaic of satellite tiles covering bounds (EPSG:3857). Returns image and its extent."""
    x0, y0, x1, y1 = bounds
    tx0, ty0 = merc_to_tile(x0, y1, z)
    tx1, ty1 = merc_to_tile(x1, y0, z)
    xs = range(int(tx0), int(tx1) + 1)
    ys = range(int(ty0), int(ty1) + 1)
    jobs = {(x, y): pool.submit(tile, x, y, z) for y in ys for x in xs}
    img = Image.new("RGB", (256 * len(xs), 256 * len(ys)))
    for (x, y), f in jobs.items():
        img.paste(f.result(), ((x - xs[0]) * 256, (y - ys[0]) * 256))
    size = 2 * math.pi * R_EARTH / 2 ** z
    left = xs[0] * size - math.pi * R_EARTH
    top = math.pi * R_EARTH - ys[0] * size
    return np.asarray(img), (left, left + size * len(xs), top - size * len(ys), top)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", action="append", required=True, help="name=path.parquet[?col=value]:color")
    p.add_argument("--zoom", type=int, default=18)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    layers = []
    for spec in a.layer:
        name, rest = spec.split("=", 1)
        path, color = rest.rsplit(":", 1)
        path, _, flt = path.partition("?")  # optional filter: path?column=value
        g = gpd.read_parquet(path)
        if flt:
            col, val = flt.split("=")
            g = g[g[col].astype(str) == val]
        g = g[["geometry"]].reset_index(drop=True)
        layers.append((name, g, color))

    views = pd.read_csv(a.views)
    to_utm = Transformer.from_crs(4326, layers[0][1].crs, always_xy=True)
    pool = ThreadPoolExecutor(32)

    for v in views.itertuples():
        cx, cy = to_utm.transform(v.lon, v.lat)
        win = box(cx - v.half_m, cy - v.half_m, cx + v.half_m, cy + v.half_m)
        win_m = gpd.GeoSeries([win], crs=layers[0][1].crs).to_crs(3857).total_bounds
        img, ext = basemap(win_m, a.zoom, pool)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        for ax in axs:
            ax.imshow(img, extent=ext)
            ax.set_xlim(win_m[0], win_m[2])
            ax.set_ylim(win_m[1], win_m[3])
            ax.set_xticks([])
            ax.set_yticks([])
        split = any(n.startswith("out") for n, _, _ in layers)  # inputs left, outputs right
        for name, g, color in layers:
            is_out = name.startswith("out")
            ax = axs[1] if (is_out or not split) else axs[0]
            sub = g.iloc[g.sindex.query(win)]
            if len(sub):
                sub = sub.clip(win.buffer(v.half_m * 0.1)).to_crs(3857)
                if is_out:
                    sub.plot(ax=ax, color=color, alpha=0.35)
                sub.boundary.plot(ax=ax, color=color, lw=1.6)
        for i, ax in enumerate(axs if split else axs[1:]):
            mine = [(n, c) for n, _, c in layers if not split or n.startswith("out") == (i == 1)]
            if mine:
                ax.legend(handles=[plt.Line2D([], [], color=c, label=n) for n, c in mine], loc="upper right")
        fig.suptitle(f"{v.name}   ({v.lat:.5f}, {v.lon:.5f})   width {2 * v.half_m:.0f} m")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{v.name}.png"), dpi=70)
        plt.close(fig)
        print("saved", v.name)


if __name__ == "__main__":
    main()
