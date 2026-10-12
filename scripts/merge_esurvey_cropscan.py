"""Merge CropScan polygons with eSurvey field boundaries.

Rules
1. eSurvey polygon with CropScan -> kept, but every empty part that is a whole
   field (inside or on the side, small or big) is cut out with straight lines
   along the field directions. The eSurvey outline is kept where CropScan comes
   within 5 m of it (edge misalignment); road / bund gaps up to ~10 m between
   two CropScan fields are kept.
2. eSurvey with no CropScan -> dropped (CropScan proves the crop).
3. CropScan with no eSurvey -> kept exactly as it is (original geometry).
4. CropScan reaching outside eSurvey -> the outside part is kept, except thin
   edge misalignment strips; small pieces get a size class for review.

Every polygon gets a `category`, `area_ac` (acres, 3 decimals) and the ids of
the eSurvey (`es_id`) and CropScan (`cs_id`) polygons it comes from.

Each eSurvey and CropScan polygon is processed in its own small window, in
parallel on every core, so run time grows linearly with the area.

Usage:
    python scripts/merge_esurvey_cropscan.py --cropscan data/cache/cropscan_utm.parquet \
        --esurvey data/cache/esurvey_utm.parquet --cropscan-original <cropscan.shp> \
        --out out/merged.gpkg [--aoi minlon minlat maxlon maxlat] [--workers N]
"""
import argparse
import multiprocessing
import os
import resource
import time

import geopandas as gpd
import numpy as np
import pyogrio
import pandas as pd
import rasterio.features as rf
import shapely
from concurrent.futures import ProcessPoolExecutor

from rasterio.transform import from_origin
from scipy import ndimage
from shapely import affinity
from shapely.geometry import box

GAP_CLOSE_M = 6.0  # closes CropScan gaps up to ~12 m wide (roads, bunds)
QUAD_SEGS = 16  # segments per quarter circle in the gap-closing buffers
HOLE_FILL_HA = 0.02  # only CropScan specks / noise holes (< 200 m2) are filled
BIG_HA = 5.0  # eSurvey polygons this big are blocks of many fields
DROP_BELOW = 0.0  # no coverage cut-off: any eSurvey with a real CropScan field is cut and kept
PIX = 2.5  # pixel size (m) for the cut grid
CELL = 2  # pixels per cell -> 5 m cells
CELL_KEEP = 0.5  # cell coverage to keep a cell
TOL_CELLS = 1  # eSurvey kept up to this many cells (5 m) from CropScan
NOISE_HA = 0.01  # empty specks smaller than this are not cut
ISLAND_HA = 0.02  # in big blocks, only CropScan specks (< 200 m2) are ignored; every real field is kept
SPILL_OPEN_M = 4.0  # only edge misalignment strips (< ~8 m wide) along eSurvey are cut off
SPILL_MIN_HA = 0.01  # CropScan outside eSurvey is kept as it is, only specks are removed
SPILL_MIN_WIDTH_M = 8.0  # thinner pieces are misalignment strips
SPILL_RECT_MIN = 0.0  # no shape filter: CropScan outside eSurvey is kept as CropScan drew it
M2_PER_ACRE = 4046.8564224
SMALL_CLASSES = [  # (upper limit in acres, category) for small CropScan pieces left beside eSurvey
    (0.10, "cropscan_small_lt_0.10ac"),
    (0.15, "cropscan_small_0.10_0.15ac"),
    (0.20, "cropscan_small_0.15_0.20ac"),
]
CATEGORY = {
    "esurvey": "esurvey_full",  # eSurvey polygon kept as it is
    "esurvey_clipped": "esurvey_clipped",  # eSurvey polygon cut to the CropScan fields
    "cropscan": "cropscan_original",  # CropScan with no eSurvey, original geometry
    "cropscan_spill": "cropscan_clipped",  # CropScan field outside eSurvey, cut at the eSurvey edge
}
MIN_PART_HA = 0.01  # output pieces smaller than this are removed


def fill_small_holes(geom, max_area):
    filled = False

    def fix(p):
        nonlocal filled
        holes = [r for r in p.interiors if shapely.Polygon(r).area >= max_area]
        filled |= len(holes) < len(p.interiors)
        return shapely.Polygon(p.exterior, holes)

    parts = [fix(p) for p in shapely.get_parts(geom) if p.geom_type == "Polygon"]
    if not parts:
        return shapely.Polygon()
    # the parts are disjoint unless a filled hole had another part inside it
    return shapely.union_all(parts) if filled else shapely.multipolygons(parts)


def dominant_angle(poly):
    """Main field direction in degrees (0-90), from the length-weighted edge angles."""
    rings = [poly.exterior] + list(poly.interiors) if poly.geom_type == "Polygon" else []
    if not rings:
        rings = [p.exterior for p in shapely.get_parts(poly)]
    xy = np.vstack([np.diff(np.asarray(r.coords), axis=0) for r in rings])
    ang = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 90
    w = np.hypot(xy[:, 0], xy[:, 1])
    hist, edges = np.histogram(ang, bins=90, range=(0, 90), weights=w)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)  # circular smoothing
    i = hist.argmax()
    sel = (ang >= edges[i] - 1) & (ang < edges[i] + 2)
    return float(np.average(ang[sel], weights=w[sel])) if sel.any() else float(edges[i])


def field_like(g):
    """Roughly rectangular and at least SPILL_MIN_WIDTH_M wide (drops wedges, triangles, blobs)."""
    r = shapely.minimum_rotated_rectangle(g)
    xy = np.asarray(r.exterior.coords)
    sides = np.hypot(*np.diff(xy[:3], axis=0).T)
    return sides.min() >= SPILL_MIN_WIDTH_M and g.area / r.area >= SPILL_RECT_MIN


def drop_islands(geom, min_area):
    parts = [g for g in shapely.get_parts(geom) if g.area >= min_area]
    return shapely.multipolygons(parts) if parts else shapely.Polygon()


def straight_cut(poly, cs_local):
    """Part of poly that holds CropScan, cut with straight lines along the field directions.

    The eSurvey outline is kept where CropScan comes within TOL_CELLS of it."""
    theta = dominant_angle(poly)
    c = poly.centroid
    rp = affinity.rotate(poly, -theta, origin=c)
    rc = affinity.rotate(cs_local, -theta, origin=c)
    x0, y0, x1, y1 = rp.bounds
    step = PIX * CELL
    w = int(np.ceil((x1 - x0) / step)) * CELL
    h = int(np.ceil((y1 - y0) / step)) * CELL
    t = from_origin(x0, y0 + h * PIX, PIX, PIX)  # grid snapped to the polygon's bottom-left corner
    pm = rf.rasterize([rp], out_shape=(h, w), transform=t, dtype="uint8")
    burn = lambda g: rf.rasterize([g], out_shape=(h, w), transform=t, dtype="uint8") if not g.is_empty else np.zeros_like(pm)
    cell = lambda a: a.reshape(h // CELL, CELL, w // CELL, CELL).sum(axis=(1, 3))
    inside = cell(pm)
    poly_cells = inside > 0
    cov = lambda m: np.where(poly_cells, cell(pm & m) / np.maximum(inside, 1), 0)
    keep = cov(burn(rc)) >= CELL_KEEP  # cells with real CropScan
    # tolerance: road / bund gaps up to 10 m between two CropScan fields are kept, and the eSurvey
    # outline is kept where CropScan is within TOL_CELLS (5 m) of it; inner cuts follow CropScan
    k = np.ones((2 * TOL_CELLS + 1,) * 2)
    near_cs = ndimage.binary_dilation(keep, k)
    keep = ndimage.binary_closing(np.pad(keep, TOL_CELLS + 1), k)[TOL_CELLS + 1:-TOL_CELLS - 1, TOL_CELLS + 1:-TOL_CELLS - 1]
    rim = poly_cells & ndimage.binary_dilation(~poly_cells, k)  # cells along the eSurvey outline
    keep = (keep | (rim & near_cs)) & poly_cells  # the outline is kept where CropScan is a few metres off it
    # tiny empty specks (< NOISE_HA) are not cut
    lab, n = ndimage.label(poly_cells & ~keep)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * step * step / 1e4
        keep |= np.isin(lab, 1 + np.flatnonzero(sizes < NOISE_HA))
    keep = ndimage.binary_opening(keep, np.ones((2, 2)))  # no kept slivers one cell (5 m) wide
    if not keep.any():
        return None
    ct = from_origin(x0, y0 + h * PIX, step, step)
    shapes = [shapely.geometry.shape(g) for g, v in rf.shapes(keep.astype("uint8"), mask=keep, transform=ct) if v]
    cut = affinity.rotate(shapely.union_all(shapes), theta, origin=c)
    return shapely.make_valid(cut.intersection(poly))


WINDOW_M = 30.0  # CropScan context around each eSurvey polygon (> 2 x GAP_CLOSE_M)
_CS = _CS_TREE = _ES = _ES_TREE = None  # set in main(), inherited by forked workers


def _snap(g, snap):
    """Inputs snapped to a 1 mm grid; used only when GEOS fails on a rare near-coincident edge."""
    return shapely.make_valid(shapely.set_precision(g, 0.001)) if snap else g


def _robust(fn, i):
    try:
        return fn(i)
    except shapely.errors.GEOSException:
        return fn(i, snap=True)


def _esurvey_job(idx):
    return [r for i in idx for r in _robust(_esurvey_job_one, i)]


def _esurvey_job_one(i, snap=False):
    """Decide one eSurvey polygon from the CropScan in a window around it."""
    rows = []
    g = _snap(_ES[i], snap)
    win = g.buffer(WINDOW_M)
    near = _CS_TREE.query(win, predicate="intersects")
    if not len(near):
        return rows
    pieces = shapely.intersection(_snap(_CS[near], snap), win)
    local = fill_small_holes(shapely.union_all(pieces), HOLE_FILL_HA * 1e4)
    close = lambda x: x.buffer(GAP_CLOSE_M, quad_segs=QUAD_SEGS).buffer(-GAP_CLOSE_M, quad_segs=QUAD_SEGS)
    ci = g.intersection(close(local))  # road / bund gaps bridged
    cv = ci.area / g.area
    if cv <= DROP_BELOW:
        return rows
    if g.area >= BIG_HA * 1e4:  # blocks of many fields: lone CropScan specks do not keep a field
        kept = drop_islands(local, ISLAND_HA * 1e4)
        if shapely.get_num_geometries(kept) < shapely.get_num_geometries(local):  # only when there are specks
            ci = g.intersection(close(kept))
    cut = None if ci.is_empty else straight_cut(g, ci)
    if cut is None or cut.is_empty:
        return rows
    if g.area - cut.area < NOISE_HA * 1e4:  # nothing cut -> original boundary
        src, cut = "esurvey", g
    else:
        src = "esurvey_clipped"
    tree = shapely.STRtree(pieces)
    for part in shapely.get_parts(cut):  # each part gets the CropScan polygon it overlaps most
        hit = tree.query(part, predicate="intersects")
        best = near[hit[np.argmax(shapely.area(shapely.intersection(pieces[hit], part)))]] if len(hit) else -1
        rows.append((part, src, cv, i, best))
    return rows


def _cropscan_job(idx):
    return [r for i in idx for r in _robust(_cropscan_job_one, i)]


def _cropscan_job_one(i, snap=False):
    """CropScan with no eSurvey stays as it is; the part of other CropScan outside eSurvey is kept
    when wider than the edge misalignment strips."""
    rows = []
    g = _snap(_CS[i], snap) if snap else _CS[i]
    near = _ES_TREE.query(g, predicate="intersects")
    if not len(near):
        rows.append((g, "cropscan", i, i))
        return rows
    spill = g.difference(shapely.union_all(_snap(_ES[near], snap)))
    # square-cornered opening: thin strips go, field corners stay as CropScan drew them
    spill = spill.buffer(-SPILL_OPEN_M, join_style="mitre").buffer(SPILL_OPEN_M, join_style="mitre").intersection(spill)
    for part in shapely.get_parts(spill):
        if part.geom_type == "Polygon" and part.area >= SPILL_MIN_HA * 1e4 and field_like(part):
            rows.append((fill_small_holes(part, HOLE_FILL_HA * 1e4), "cropscan_spill", i, i))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cropscan", required=True)
    p.add_argument("--esurvey", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cropscan-original", help="original CropScan file (row order = cropscan ids); "
                   "untouched CropScan polygons are written with its exact geometry")
    p.add_argument("--workers", type=int, help="processes (default: all cores)")
    p.add_argument("--aoi", nargs=4, type=float, help="minlon minlat maxlon maxlat")
    a = p.parse_args()
    t0 = time.time()

    cs = gpd.read_parquet(a.cropscan)
    es = gpd.read_parquet(a.esurvey)
    # subsets carry the original ids in a column, keep them as the index
    cs = cs.set_index("cropscan_id") if "cropscan_id" in cs else cs
    es = es.set_index("esurvey_id") if "esurvey_id" in es else es
    cs, es = cs[["geometry"]], es[["geometry"]]
    crs = cs.crs
    if a.aoi:
        aoi = gpd.GeoSeries([box(*a.aoi)], crs=4326).to_crs(crs).iloc[0]
        es = es.iloc[es.sindex.query(aoi)]
        work = box(*es.total_bounds).union(aoi).envelope.buffer(200)  # whole eSurvey parts + margin
        cs = cs.iloc[cs.sindex.query(work)]
        cs = cs.set_geometry(shapely.intersection(cs.geometry.values, work))
    else:
        aoi = None
    print(f"loaded cs {len(cs)}  es {len(es)}  {time.time() - t0:.1f}s")

    # each eSurvey polygon and each CropScan polygon is worked on in its own small window,
    # in parallel on every core (forked workers share the inputs read-only)
    global _CS, _CS_TREE, _ES, _ES_TREE
    _CS, _ES = cs.geometry.values, es.geometry.values
    _CS_TREE, _ES_TREE = shapely.STRtree(_CS), shapely.STRtree(_ES)
    workers = a.workers or os.cpu_count()
    ctx = multiprocessing.get_context("fork")
    out = []
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        # biggest polygons first, handed out a few at a time, so no worker is left with all the big ones
        order = np.argsort(-shapely.area(_ES))
        for rows in pool.map(_esurvey_job, [order[k:k + 4] for k in range(0, len(order), 4)]):
            out += [(g, src, cv, es.index[i], cs.index[c] if c >= 0 else -1) for g, src, cv, i, c in rows]
        print(f"esurvey decided  {time.time() - t0:.1f}s")
        order = np.argsort(-shapely.get_num_coordinates(_CS))
        for rows in pool.map(_cropscan_job, [order[k:k + 16] for k in range(0, len(order), 16)]):
            out += [(g, src, np.nan, -1, cs.index[c]) for g, src, i, c in rows]
    print(f"cropscan done  {time.time() - t0:.1f}s")

    res = gpd.GeoDataFrame(out, columns=["geometry", "source", "es_cover", "es_id", "cs_id"], crs=crs)
    res = res.explode(index_parts=False)
    res = res[res.geom_type == "Polygon"]
    if aoi is not None:
        res = res[res.intersects(aoi)]
    # untouched CropScan is never filtered; new pieces below MIN_PART_HA are dropped
    res = res[(res.source == "cropscan") | (res.area >= MIN_PART_HA * 1e4)].reset_index(drop=True)
    res.insert(0, "fid", np.arange(1, len(res) + 1))
    res.insert(1, "predicted", 1)  # Sugarcane
    res["es_cover"] = res.es_cover.round(3)
    res["area_ac"] = (res.area / M2_PER_ACRE).round(3)
    # review flag: small or narrow CropScan pieces left beside eSurvey after the clip
    # category: one class per polygon, for styling and review
    cat = res.source.map(CATEGORY)
    # small CropScan pieces beside eSurvey get a size class; bigger clipped fields keep cropscan_clipped
    spill = res.source == "cropscan_spill"
    for upper, name in SMALL_CLASSES[::-1]:
        cat = cat.where(~(spill & (res.area_ac < upper)), name)
    res["category"] = cat
    print(f"attributes  {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_parquet(os.path.splitext(a.out)[0] + ".parquet")
    out4326 = res.to_crs(4326)
    if a.cropscan_original:  # untouched CropScan: the exact original geometry, every vertex as delivered
        keep = out4326.index[out4326.source == "cropscan"]
        ids = out4326.cs_id[keep].to_numpy()
        orig = pyogrio.read_dataframe(a.cropscan_original, fids=ids, columns=[])
        same = np.isclose(orig.to_crs(crs).area.to_numpy(), res.area[keep].to_numpy(), rtol=1e-6)
        out4326.loc[keep[same], "geometry"] = orig.geometry.values[same]  # parts cut by an AOI stay cut
        print(f"original geometry restored for {same.sum()} of {len(keep)} untouched CropScan polygons")
    print(f"originals restored  {time.time() - t0:.1f}s")
    out4326.to_file(a.out, driver="GPKG", engine="pyogrio")
    print(res.groupby("source").area_ac.agg(["count", "sum"]).round(1))
    print(res.groupby("category").area_ac.agg(["count", "sum"]).round(1))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved {a.out}  {time.time() - t0:.1f}s  peak memory {peak:.1f} GB")


if __name__ == "__main__":
    main()
