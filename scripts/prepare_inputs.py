"""Convert the CropScan and eSurvey files into GeoParquet files in a metric (UTM) CRS.

CropScan keeps one row per input feature; its feature id (row number in the
original file) is the index, i.e. the CropScan id (`cs_id`) in the output.
eSurvey multipolygons are split into single polygons and invalid rings are
repaired; its own `Id` field, if present, is kept as `esurvey_Id`.

With --aoi only the polygons that lie in the AOI (by an interior point) are
kept. Polygons are never cut at the AOI line, so every polygon belongs to
exactly one AOI and CropScan geometry stays as delivered.

Files that already exist are not rebuilt (delete them to force it); files are
written under a temporary name first, so a stopped run never leaves a broken file.

Usage:
    python scripts/prepare_inputs.py --cropscan <cropscan.shp> --esurvey <esurvey.shp> \
        [--aoi <aoi.shp>] --out data/cache/<name>
Writes <out>/cropscan_utm.parquet, <out>/esurvey_utm.parquet and <out>/cropscan_original.parquet
(the CropScan geometry exactly as delivered, used to write untouched CropScan polygons).
"""
import argparse
import os
import time

import geopandas as gpd
import shapely


def in_aoi(g, aoi):
    if aoi is None:
        return g
    return g[shapely.within(shapely.point_on_surface(g.geometry.values), aoi)]


def save(g, path):
    g.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cropscan", required=True)
    p.add_argument("--esurvey", required=True)
    p.add_argument("--aoi", help="AOI polygon file; only polygons inside it are kept")
    p.add_argument("--out", default="data/cache")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    aoi = mask = None
    if a.aoi:
        aoi_df = gpd.read_file(a.aoi, engine="pyogrio").to_crs(4326)
        aoi = shapely.union_all(aoi_df.geometry.values)
        mask = aoi
        shapely.prepare(aoi)

    cs_path = os.path.join(a.out, "cropscan_utm.parquet")
    es_path = os.path.join(a.out, "esurvey_utm.parquet")
    utm = None
    if os.path.exists(cs_path):
        utm = gpd.read_parquet(cs_path, columns=["geometry"]).crs
        print(f"cropscan: already prepared ({cs_path})")
    else:
        cs = gpd.read_file(a.cropscan, engine="pyogrio", use_arrow=True, mask=mask, fid_as_index=True,
                           columns=[])
        if cs.crs.to_epsg() != 4326:
            raise SystemExit("CropScan is expected in EPSG:4326 so its original geometry can be delivered as is")
        cs = in_aoi(cs, aoi)
        utm = cs.estimate_utm_crs()
        # the original geometry, unchanged (same coordinates and vertices), for the delivery file
        save(cs[["geometry"]], os.path.join(a.out, "cropscan_original.parquet"))
        save(cs[["geometry"]].to_crs(utm), cs_path)
        print(f"cropscan: {len(cs)} polygons -> {utm.to_string()}  {time.time() - t0:.0f}s")

    if os.path.exists(es_path):
        print(f"esurvey: already prepared ({es_path})")
    else:
        es = gpd.read_file(a.esurvey, engine="pyogrio", use_arrow=True, mask=mask).to_crs(4326)
        es = in_aoi(es, aoi)
        es = es.rename(columns={"Id": "esurvey_Id"})
        es = es[[c for c in ["esurvey_Id", "geometry"] if c in es]].explode(index_parts=False)
        es["geometry"] = shapely.make_valid(es.geometry.values, method="structure")
        es = es.explode(index_parts=False)
        es = es[es.geom_type == "Polygon"].to_crs(utm).reset_index(drop=True)
        save(es, es_path)
        print(f"esurvey: {len(es)} polygons  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
