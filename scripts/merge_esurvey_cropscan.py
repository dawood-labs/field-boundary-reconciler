"""Merge CropScan sugarcane with eSurvey field boundaries.

Rules
1. eSurvey polygon (almost) filled by CropScan, ignoring the thin road / bund
   gaps between fields -> keep the eSurvey polygon as it is.
2. CropScan with no eSurvey -> keep the CropScan polygon as it is.
3. eSurvey with no CropScan -> drop.
4. eSurvey partly covered -> every empty part that is a whole field (small or
   big) is cut off with straight lines along the field directions; the eSurvey
   boundary is kept only where it is a few metres off the CropScan edge.
5. CropScan spilling outside eSurvey -> thin spills are cut off, field-shaped
   spills are kept as separate polygons.

Inputs are the cached GeoParquet files in a metric CRS (see README).

Usage:
    python scripts/merge_esurvey_cropscan.py --cropscan data/cache/cropscan_utm.parquet \
        --esurvey data/cache/esurvey_utm.parquet --out out/merged.gpkg \
        [--aoi minlon minlat maxlon maxlat]
"""
import argparse
import os
import resource
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features as rf
import shapely
from rasterio.transform import from_origin
from scipy import ndimage
from shapely import affinity
from shapely.geometry import box

GAP_CLOSE_M = 6.0  # closes CropScan gaps up to ~12 m wide (roads, bunds)
HOLE_FILL_HA = 0.1  # CropScan holes smaller than this are filled
BIG_HA = 5.0  # eSurvey polygons this big are blocks of many fields
DROP_BELOW = 0.05  # eSurvey coverage below this -> drop
PIX = 2.0  # pixel size (m) for the cut grid
CELL = 5  # pixels per cell -> 10 m cells
CELL_KEEP = 0.5  # cell coverage to keep a cell
EDGE_M = 8.0  # CropScan sits a few metres inside field edges -> grown by this before cutting
CUT_MIN_CELLS = 2  # a part is cut off only if it is at least this many cells (20 m) wide ...
CUT_MIN_HA = 0.1  # ... and at least this big: any whole field without CropScan is cut, small or big
ISLAND_HA = 0.15  # in big blocks, lone CropScan bits smaller than this do not keep a field
SPILL_OPEN_M = 7.5  # spills narrower than ~15 m (edge misalignment strips) are cut off
SPILL_MIN_HA = 0.05  # smaller spills are cut off; anything bigger and wide enough is a field and kept
SPILL_MIN_WIDTH_M = 15.0  # spills narrower than this (rotated box) are cut off
SPILL_RECT_MIN = 0.0  # shape and basemap checks are done later by check_spills.py
MIN_PART_HA = 0.05  # output pieces smaller than this are removed


def fill_small_holes(geom, max_area):
    def fix(p):
        holes = [r for r in p.interiors if shapely.Polygon(r).area >= max_area]
        return shapely.Polygon(p.exterior, holes)

    parts = shapely.get_parts(geom)
    return shapely.union_all([fix(p) for p in parts if p.geom_type == "Polygon"])


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
    return shapely.union_all(parts) if parts else shapely.Polygon()


def straight_cut(poly, cs_local, cs_grown):
    """Part of poly that holds CropScan, cut with straight lines along the field directions.

    Empty parts are found with cs_grown (CropScan grown over the edge gap), then the cut
    is moved out to the real CropScan edge (cs_local)."""
    theta = dominant_angle(poly)
    c = poly.centroid
    rp = affinity.rotate(poly, -theta, origin=c)
    rc = affinity.rotate(cs_local, -theta, origin=c)
    rg = affinity.rotate(cs_grown, -theta, origin=c)
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
    raw_empty = poly_cells & (cov(burn(rc)) < CELL_KEEP)
    keep = cov(burn(rg)) >= CELL_KEEP
    # empty parts are cut off only when they are field-sized: wide enough and big enough
    empty = poly_cells & ~keep
    empty = ndimage.binary_opening(empty, np.ones((CUT_MIN_CELLS, CUT_MIN_CELLS)))
    lab, n = ndimage.label(empty)
    if n:
        sizes = ndimage.sum(empty, lab, range(1, n + 1)) * step * step / 1e4
        empty = np.isin(lab, 1 + np.flatnonzero(sizes >= CUT_MIN_HA))
    empty |= ndimage.binary_dilation(empty, np.ones((3, 3)), iterations=1) & raw_empty  # cut at the real edge
    keep = poly_cells & ~empty
    keep = ndimage.binary_closing(np.pad(keep, 3), np.ones((3, 3)))[3:-3, 3:-3] & poly_cells  # fill notches up to 30 m
    keep = ndimage.binary_opening(keep, np.ones((CUT_MIN_CELLS, CUT_MIN_CELLS)))  # no thin leftovers
    keep = ndimage.binary_fill_holes(keep)  # never cut out the centre
    if not keep.any():
        return None
    ct = from_origin(x0, y0 + h * PIX, step, step)
    shapes = [shapely.geometry.shape(g) for g, v in rf.shapes(keep.astype("uint8"), mask=keep, transform=ct) if v]
    cut = affinity.rotate(shapely.union_all(shapes), theta, origin=c)
    return shapely.make_valid(cut.intersection(poly))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cropscan", required=True)
    p.add_argument("--esurvey", required=True)
    p.add_argument("--out", required=True)
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

    # CropScan cleaned (small holes filled) and closed (road / bund gaps bridged)
    cs_raw = shapely.union_all(cs.geometry.values)
    cs_clean = fill_small_holes(cs_raw, HOLE_FILL_HA * 1e4)
    cs_closed = cs_clean.buffer(GAP_CLOSE_M).buffer(-GAP_CLOSE_M)
    shapely.prepare(cs_closed)
    # for big blocks: lone small CropScan polygons are removed before the gaps are closed
    cs_big = drop_islands(cs_clean, ISLAND_HA * 1e4).buffer(GAP_CLOSE_M).buffer(-GAP_CLOSE_M)
    print(f"cropscan cleaned  {time.time() - t0:.1f}s")

    # eSurvey decisions
    out = []
    es_geoms = es.geometry.values
    inter = shapely.intersection(es_geoms, cs_closed)
    cov = shapely.area(inter) / shapely.area(es_geoms)
    for eid, g, cv, ci in zip(es.index, es_geoms, cov, inter):
        if cv < DROP_BELOW:
            continue
        if g.area >= BIG_HA * 1e4:  # blocks of many fields: lone tiny CropScan bits do not keep a field
            ci = g.intersection(cs_big)
        cut = None if ci.is_empty else straight_cut(g, ci, ci.buffer(EDGE_M).intersection(g))
        if cut is None or cut.is_empty:
            continue
        if cut.area >= 0.98 * g.area:  # nothing worth cutting -> original boundary
            out.append((g, "esurvey", cv, eid))
        else:
            out.append((cut, "esurvey_clipped", cv, eid))
    print(f"esurvey decided  {time.time() - t0:.1f}s")

    # CropScan outside eSurvey: whole polygons with no eSurvey stay as they are,
    # spills from polygons that touch eSurvey are kept only when field-shaped
    es_all = shapely.union_all(es_geoms)
    shapely.prepare(es_all)
    touches = shapely.intersects(cs.geometry.values, es_all)
    for g in cs.geometry.values[~touches]:
        out.append((fill_small_holes(g, HOLE_FILL_HA * 1e4), "cropscan", np.nan, -1))
    spill = shapely.difference(shapely.union_all(cs.geometry.values[touches]), es_all)
    # square-cornered opening: thin strips go, field corners stay as CropScan drew them
    spill = spill.buffer(-SPILL_OPEN_M, join_style="mitre").buffer(SPILL_OPEN_M, join_style="mitre").intersection(spill)
    for g in shapely.get_parts(spill):
        if g.area >= SPILL_MIN_HA * 1e4 and field_like(g):
            out.append((fill_small_holes(g, HOLE_FILL_HA * 1e4), "cropscan_spill", np.nan, -1))
    print(f"cropscan done  {time.time() - t0:.1f}s")

    res = gpd.GeoDataFrame(out, columns=["geometry", "source", "es_cover", "es_id"], crs=crs)
    res = res.explode(index_parts=False)
    res = res[res.geom_type == "Polygon"]
    if aoi is not None:
        res = res[res.intersects(aoi)]
    res = res[res.area >= MIN_PART_HA * 1e4].reset_index(drop=True)
    # cs_id: the CropScan polygon that overlaps each output polygon most (-1 if none)
    cs_ids = cs.index.to_numpy()  # ids can repeat in a subset (one polygon cut into parts)
    cs_pos = cs.reset_index(drop=True)
    pairs = gpd.sjoin(res[["geometry"]], cs_pos[["geometry"]], predicate="intersects").reset_index()
    pairs["ov"] = shapely.area(shapely.intersection(res.geometry.values[pairs["index"]], cs_pos.geometry.values[pairs.index_right]))
    best = pairs.sort_values("ov").drop_duplicates("index", keep="last").set_index("index").index_right
    res["cs_id"] = -1
    res.loc[best.index, "cs_id"] = cs_ids[best.to_numpy()]
    res.insert(0, "fid", np.arange(1, len(res) + 1))
    res.insert(1, "predicted", 1)  # Sugarcane
    res["area_ha"] = (res.area / 1e4).round(4)
    res["es_cover"] = res.es_cover.round(3)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_parquet(os.path.splitext(a.out)[0] + ".parquet")
    res.to_crs(4326).to_file(a.out, driver="GPKG", engine="pyogrio")
    print(res.groupby("source").area_ha.agg(["count", "sum"]).round(1))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved {a.out}  {time.time() - t0:.1f}s  peak memory {peak:.1f} GB")


if __name__ == "__main__":
    main()
