"""Download satellite basemap tiles once and keep them on disk.

Only tiles that touch the given layers (plus a margin) are fetched. Tiles that
are already on disk are skipped, so a stopped run continues where it left off.
Tiles are saved as <out>/<zoom>/<x>/<y>.jpg; render_views.py and
check_spills.py read them from there.

Usage:
    python scripts/cache_tiles.py --layer out/subset/cropscan_subset.parquet \
        --layer out/subset/esurvey_subset.parquet --out data/tiles --zoom 18
"""
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import requests
import shapely

TILE_URL = "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
R_EARTH = 6378137.0
WORKERS = 32  # parallel network streams


def tile_path(root, z, x, y):
    return os.path.join(root, str(z), str(x), f"{y}.jpg")


def tiles_for(geom_3857, z):
    """(x, y) of all tiles at zoom z that touch geom (EPSG:3857)."""
    size = 2 * math.pi * R_EARTH / 2 ** z
    x0, y0, x1, y1 = geom_3857.bounds
    cols = range(int((x0 + math.pi * R_EARTH) // size), int((x1 + math.pi * R_EARTH) // size) + 1)
    rows = range(int((math.pi * R_EARTH - y1) // size), int((math.pi * R_EARTH - y0) // size) + 1)
    boxes, keys = [], []
    for x in cols:
        for y in rows:
            left, top = x * size - math.pi * R_EARTH, math.pi * R_EARTH - y * size
            boxes.append(shapely.box(left, top - size, left + size, top))
            keys.append((x, y))
    hit = shapely.intersects(boxes, geom_3857)
    return [k for k, h in zip(keys, hit) if h]


def fetch(session, root, z, x, y):
    dst = tile_path(root, z, x, y)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    r = session.get(TILE_URL.format(s=(x + y) % 4, x=x, y=y, z=z), timeout=30)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dst)  # never leave a half-written tile
    return len(r.content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", action="append", required=True, help="GeoParquet layer(s) to cover")
    p.add_argument("--out", default="data/tiles")
    p.add_argument("--zoom", type=int, default=18)
    p.add_argument("--margin-m", type=float, default=100.0)
    a = p.parse_args()

    geoms = []
    for path in a.layer:
        g = gpd.read_parquet(path, columns=["geometry"])
        geoms.append(shapely.union_all(g.geometry.buffer(a.margin_m).values))
        area = gpd.GeoSeries([geoms[-1]], crs=g.crs).to_crs(3857).iloc[0]
        geoms[-1] = area
    cover = shapely.union_all(geoms)
    keys = tiles_for(cover, a.zoom)
    todo = [k for k in keys if not os.path.exists(tile_path(a.out, a.zoom, *k))]
    print(f"tiles needed {len(keys)}, already cached {len(keys) - len(todo)}, to download {len(todo)}")

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))
    done = size = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for n in pool.map(lambda k: fetch(session, a.out, a.zoom, *k), todo):
            done += 1
            size += n
            if done % 2000 == 0:
                print(f"  {done}/{len(todo)}  {size / 1e6:.0f} MB")
    print(f"done: {done} downloaded, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
