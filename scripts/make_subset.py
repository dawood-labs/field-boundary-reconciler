"""Clip the cached CropScan and eSurvey layers to an AOI, for quick test runs.

Writes, in --out:
    aoi.gpkg                 the AOI rectangle
    cropscan_subset.parquet / .gpkg
    esurvey_subset.parquet / .gpkg
The .parquet files (metric CRS) are the inputs for merge_esurvey_cropscan.py,
the .gpkg files (EPSG:4326) are for viewing in QGIS.

Usage:
    python scripts/make_subset.py --cropscan data/cache/cropscan_utm.parquet \
        --esurvey data/cache/esurvey_utm.parquet --aoi minlon minlat maxlon maxlat --out out/subset
"""
import argparse
import os

import geopandas as gpd
import shapely
from shapely.geometry import box


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cropscan", required=True)
    p.add_argument("--esurvey", required=True)
    p.add_argument("--aoi", nargs=4, type=float, required=True, help="minlon minlat maxlon maxlat")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    aoi = gpd.GeoDataFrame(geometry=[box(*a.aoi)], crs=4326)
    aoi.to_file(os.path.join(a.out, "aoi.gpkg"), driver="GPKG", engine="pyogrio")

    for name, path in [("cropscan", a.cropscan), ("esurvey", a.esurvey)]:
        g = gpd.read_parquet(path)
        win = aoi.to_crs(g.crs).geometry.iloc[0]
        g = g.iloc[g.sindex.query(win)]
        g = g.set_geometry(shapely.intersection(g.geometry.values, win))
        g = g.explode(index_parts=False)
        g = g[g.geom_type == "Polygon"]
        g = g[["geometry"]].rename_axis(f"{name}_id").reset_index()  # keeps the original ids
        g["area_ha"] = (g.area / 1e4).round(4)
        g.to_parquet(os.path.join(a.out, f"{name}_subset.parquet"))
        g.to_crs(4326).to_file(os.path.join(a.out, f"{name}_subset.gpkg"), driver="GPKG", engine="pyogrio")
        print(f"{name}: {len(g)} polygons  {g.area_ha.sum():.1f} ha")


if __name__ == "__main__":
    main()
