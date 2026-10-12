"""Decide which CropScan spills (CropScan outside eSurvey) are real fields, using shape and the basemap.

Steps (each writes a file, so a stopped run can go on from the last step):
  features  shape + basemap features for every spill            -> <out>/spill_features.parquet
  grids     numbered basemap chips (16 per image) for labelling  -> <out>/grids/grid_XX.png
  fit       fit a small decision tree on labels (fid,label)      -> <out>/spill_model.pkl
  apply     keep only spills the model calls a field             -> <result>_checked.gpkg / .parquet

Usage:
    python scripts/check_spills.py features --result out/subset/result_subset.parquet \
        --esurvey out/subset/esurvey_subset.parquet --out out/subset/spills
    python scripts/check_spills.py grids --result ... --esurvey ... --out out/subset/spills
    python scripts/check_spills.py fit --out out/subset/spills --labels out/subset/spills/labels.csv
    python scripts/check_spills.py apply --result ... --out out/subset/spills
Basemap tiles come from the cache written by cache_tiles.py (--tiles, zoom 18).
"""
import argparse
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features as rf
import shapely
from PIL import Image
from rasterio.transform import from_bounds
from scipy import ndimage

R_EARTH = 6378137.0
ZOOM = 18
PAD_M = 30.0  # context around a spill
RING_M = 12.0  # outside ring used for contrast
EDGE_M = 3.0  # band around the boundary used for edge strength
KEEP_RECALL = 0.97  # at least this share of real fields must be kept
FEATURES = ["area_ha", "width_m", "rect_ratio", "convexity", "elong", "es_edge_share",
            "in_std", "ring_contrast", "edge_ratio", "in_green", "in_dark"]


def mosaic(bounds, tiles):
    """RGB array and its bounds (EPSG:3857) from cached tiles covering bounds."""
    size = 2 * math.pi * R_EARTH / 2 ** ZOOM
    x0, y0, x1, y1 = bounds
    cols = range(int((x0 + math.pi * R_EARTH) // size), int((x1 + math.pi * R_EARTH) // size) + 1)
    rows = range(int((math.pi * R_EARTH - y1) // size), int((math.pi * R_EARTH - y0) // size) + 1)
    img = np.zeros((256 * len(rows), 256 * len(cols), 3), np.uint8)
    for j, y in enumerate(rows):
        for i, x in enumerate(cols):
            path = os.path.join(tiles, str(ZOOM), str(x), f"{y}.jpg")
            if os.path.exists(path):
                img[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(Image.open(path).convert("RGB"))
    left = cols[0] * size - math.pi * R_EARTH
    top = math.pi * R_EARTH - rows[0] * size
    return img, (left, top - size * len(rows), left + size * len(cols), top)


def shape_features(g, es_band):
    r = shapely.minimum_rotated_rectangle(g)
    xy = np.asarray(r.exterior.coords)
    sides = np.hypot(*np.diff(xy[:3], axis=0).T)
    on_es = g.boundary.intersection(es_band).length / g.boundary.length
    return dict(area_ha=g.area / 1e4, width_m=sides.min(), rect_ratio=g.area / r.area,
                convexity=g.area / g.convex_hull.area, elong=sides.max() / max(sides.min(), 1e-6),
                es_edge_share=on_es)


def image_features(g_3857, tiles):
    b = g_3857.buffer(PAD_M).bounds
    img, mb = mosaic(b, tiles)
    h, w = img.shape[:2]
    t = from_bounds(*mb, w, h)
    m_in = rf.rasterize([g_3857.buffer(-EDGE_M)], out_shape=(h, w), transform=t, dtype="uint8").astype(bool)
    m_poly = rf.rasterize([g_3857], out_shape=(h, w), transform=t, dtype="uint8").astype(bool)
    if m_in.sum() < 20:
        m_in = m_poly
    ring = rf.rasterize([g_3857.buffer(RING_M).difference(g_3857.buffer(EDGE_M))], out_shape=(h, w),
                        transform=t, dtype="uint8").astype(bool)
    band = rf.rasterize([g_3857.boundary.buffer(EDGE_M)], out_shape=(h, w), transform=t, dtype="uint8").astype(bool)
    f = img.astype(np.float32)
    gray = f.mean(axis=2)
    grad = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    inside, out = f[m_in], f[ring] if ring.any() else f[m_in]
    r, gch, bl = inside[:, 0], inside[:, 1], inside[:, 2]
    return dict(in_std=float(inside.std(axis=0).mean()),
                ring_contrast=float(np.abs(inside.mean(axis=0) - out.mean(axis=0)).mean()),
                edge_ratio=float(grad[band].mean() / max(grad[m_in].mean(), 1e-6)),
                in_green=float(((gch - r) / np.maximum(gch + r, 1)).mean()),
                in_dark=float((inside.mean(axis=1) < 60).mean()))


def load_spills(result):
    res = gpd.read_parquet(result)
    return res[res.source == "cropscan_spill"]


def cmd_features(a):
    sp = load_spills(a.result)
    es = gpd.read_parquet(a.esurvey, columns=["geometry"])
    es_band = shapely.union_all(es.boundary.values).buffer(1.0)  # built once: eSurvey edges +-1 m
    shapely.prepare(es_band)
    g3857 = sp.to_crs(3857).geometry.values
    with ProcessPoolExecutor(os.cpu_count()) as pool:  # one process per core, ~0.2 GB each
        img = list(pool.map(partial(image_features, tiles=a.tiles), g3857, chunksize=16))
    rows = [{"fid": f, **shape_features(g, es_band), **im} for f, g, im in zip(sp.fid, sp.geometry.values, img)]
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_parquet(os.path.join(a.out, "spill_features.parquet"))
    print(f"features for {len(df)} spills -> {a.out}/spill_features.parquet")


def cmd_grids(a):
    sp = load_spills(a.result).to_crs(3857)
    if a.fids:  # only these spills, e.g. disagreements to check by eye
        sp = sp[sp.fid.isin([int(f) for f in a.fids.split(",")])]
    es = gpd.read_parquet(a.esurvey, columns=["geometry"]).to_crs(3857)
    res = gpd.read_parquet(a.result).to_crs(3857)
    os.makedirs(os.path.join(a.out, "grids"), exist_ok=True)
    per = 16
    for k in range(0, len(sp), per):
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        for ax, (_, s) in zip(axs.ravel(), sp.iloc[k:k + per].iterrows()):
            g = s.geometry
            cx, cy = g.centroid.x, g.centroid.y
            half = max(60.0, 0.75 * max(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1]))
            win = (cx - half, cy - half, cx + half, cy + half)
            img, mb = mosaic(win, a.tiles)
            ax.imshow(img, extent=(mb[0], mb[2], mb[1], mb[3]))
            win_box = shapely.box(*win)
            es.iloc[es.sindex.query(win_box)].boundary.plot(ax=ax, color="red", lw=1)
            other = res.iloc[res.sindex.query(win_box)]
            other[other.source != "cropscan_spill"].boundary.plot(ax=ax, color="cyan", lw=0.8)
            gpd.GeoSeries([g]).boundary.plot(ax=ax, color="orange", lw=2.2)
            ax.set_xlim(win[0], win[2])
            ax.set_ylim(win[1], win[3])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"fid {s.fid}", fontsize=13)
        for ax in axs.ravel()[len(sp.iloc[k:k + per]):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "grids", f"{a.prefix}_{k // per:02d}.png"), dpi=45)
        plt.close(fig)
    print(f"{math.ceil(len(sp) / per)} grids -> {a.out}/grids")


def cmd_fit(a):
    """Small decision tree; a spill is dropped only when the tree is sure (fields are kept first)."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.tree import DecisionTreeClassifier, export_text

    df = pd.read_parquet(os.path.join(a.out, "spill_features.parquet"))
    d = df.merge(pd.read_csv(a.labels), on="fid")
    y = (d.label.str.lower() == "keep").astype(int).to_numpy()
    best = None
    for depth in (2, 3, 4):
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=0)
        p_keep = cross_val_predict(clf, d[FEATURES], y, cv=5, method="predict_proba")[:, 1]
        for thr in np.arange(0.05, 0.55, 0.05):  # drop only when P(keep) < thr
            pred = (p_keep >= thr).astype(int)
            keep_recall = (pred[y == 1] == 1).mean()
            drop_recall = (pred[y == 0] == 0).mean()
            if keep_recall >= KEEP_RECALL and (best is None or drop_recall > best[0]):
                best = (drop_recall, keep_recall, depth, thr)
    drop_recall, keep_recall, depth, thr = best
    print(f"chosen: depth {depth}, drop when P(keep) < {thr:.2f}; cross-validated: "
          f"fields kept {keep_recall:.1%}, non-fields dropped {drop_recall:.1%}")
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=0).fit(d[FEATURES], y)
    print(export_text(clf, feature_names=FEATURES))
    with open(os.path.join(a.out, "spill_model.pkl"), "wb") as f:
        pickle.dump({"tree": clf, "threshold": thr}, f)


def cmd_apply(a):
    df = pd.read_parquet(os.path.join(a.out, "spill_features.parquet"))
    with open(os.path.join(a.out, "spill_model.pkl"), "rb") as f:
        m = pickle.load(f)
    drop = set(df.fid[m["tree"].predict_proba(df[FEATURES])[:, 1] < m["threshold"]])
    res = gpd.read_parquet(a.result)
    res = res[~res.fid.isin(drop)].reset_index(drop=True)
    base = os.path.splitext(a.result)[0] + "_checked"
    res.to_parquet(base + ".parquet")
    res.to_crs(4326).to_file(base + ".gpkg", driver="GPKG", engine="pyogrio")
    print(f"dropped {len(drop)} of {len(df)} spills -> {base}.gpkg")
    print(res.groupby("source").area_ha.agg(["count", "sum"]).round(1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["features", "grids", "fit", "apply"])
    p.add_argument("--result")
    p.add_argument("--esurvey")
    p.add_argument("--out", required=True)
    p.add_argument("--labels")
    p.add_argument("--tiles", default="data/tiles")
    p.add_argument("--fids", help="grids: comma-separated fids to show")
    p.add_argument("--prefix", default="grid", help="grids: file name prefix")
    a = p.parse_args()
    {"features": cmd_features, "grids": cmd_grids, "fit": cmd_fit, "apply": cmd_apply}[a.step](a)


if __name__ == "__main__":
    main()
