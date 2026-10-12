"""Convert the CropScan and eSurvey shapefiles into GeoParquet files in a metric (UTM) CRS.

CropScan keeps one row per input feature; its row number is the CropScan id
(`cs_id` in the output) and matches the feature order of the original file.
eSurvey multipolygons are split into single polygons and invalid rings are
repaired.

Usage:
    python scripts/prepare_inputs.py --cropscan <cropscan.shp> --esurvey <esurvey.shp> --out data/cache
Writes data/cache/cropscan_utm.parquet and data/cache/esurvey_utm.parquet.
"""
import argparse
import os

import geopandas as gpd
import shapely


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cropscan", required=True)
    p.add_argument("--esurvey", required=True)
    p.add_argument("--out", default="data/cache")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cs = gpd.read_file(a.cropscan, engine="pyogrio", use_arrow=True)
    utm = cs.estimate_utm_crs()
    cs = cs[["geometry"]].to_crs(utm)
    cs.to_parquet(os.path.join(a.out, "cropscan_utm.parquet"))
    print(f"cropscan: {len(cs)} polygons -> {utm.to_string()}")

    es = gpd.read_file(a.esurvey, engine="pyogrio", use_arrow=True)[["geometry"]]
    es = es.explode(index_parts=False)
    es["geometry"] = shapely.make_valid(es.geometry.values, method="structure")
    es = es.explode(index_parts=False)
    es = es[es.geom_type == "Polygon"].to_crs(utm).reset_index(drop=True)
    es.to_parquet(os.path.join(a.out, "esurvey_utm.parquet"))
    print(f"esurvey: {len(es)} polygons")


if __name__ == "__main__":
    main()
