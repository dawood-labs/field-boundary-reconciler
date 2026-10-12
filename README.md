# field-boundary-reconciler

Turns model-detected crop polygons (CropScan) into polygons that follow surveyed
field boundaries (eSurvey). Where both exist, the output keeps the surveyed
boundary but cuts away every field that has no detected crop. Where only the
model found the crop, the model polygon is kept exactly as it is.

## Rules

| Case | Output | `category` |
|---|---|---|
| eSurvey polygon fully covered by CropScan (thin road / bund gaps ignored) | eSurvey polygon as it is | `esurvey_full` |
| eSurvey polygon partly covered | eSurvey cut with straight lines along the field directions; every empty field (inside or on the side, small or big) is removed. The outline stays where CropScan is within 5 m of it. | `esurvey_clipped` |
| eSurvey polygon with no CropScan | dropped | – |
| CropScan polygon with no eSurvey | original CropScan geometry, unchanged (same vertices) | `cropscan_original` |
| Part of a CropScan polygon outside eSurvey | kept, except edge misalignment strips narrower than ~8 m | `cropscan_clipped` |
| Same, but smaller than 0.20 acres | kept, flagged by size for review | `cropscan_small_lt_0.10ac`, `cropscan_small_0.10_0.15ac`, `cropscan_small_0.15_0.20ac` |

Output fields: `fid`, `predicted` (1 = Sugarcane), `source`, `category`,
`es_cover` (share of the eSurvey polygon covered by CropScan), `es_id`,
`cs_id` (row number of the source polygons, -1 if none), `area_ac`
(acres, 3 decimals).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, any machine. The merge uses every core; memory stays below
~3 GB for ~24,000 CropScan and ~10,000 eSurvey polygons.

## Steps

All paths below are examples; keep data, outputs and credentials out of git
(`data/`, `out/`, `qa/`, `*.json`, `*.zip` are ignored).

1. **Download the inputs** (zipped shapefiles) from Google Cloud Storage:

   ```bash
   python scripts/download_inputs.py --bucket <bucket> --prefix <prefix/> \
       --key <service_account.json> --out data/raw
   unzip data/raw/<cropscan>.zip -d data/extracted/cropscan
   unzip data/raw/<esurvey>.zip -d data/extracted/esurvey
   ```

2. **Prepare the inputs** (UTM GeoParquet, eSurvey split into single polygons and repaired):

   ```bash
   python scripts/prepare_inputs.py --cropscan data/extracted/cropscan/<cropscan>.shp \
       --esurvey data/extracted/esurvey/<esurvey>.shp --out data/cache
   ```

3. **Optional: make a small test area** to try rule changes quickly:

   ```bash
   python scripts/make_subset.py --cropscan data/cache/cropscan_utm.parquet \
       --esurvey data/cache/esurvey_utm.parquet --aoi <minlon> <minlat> <maxlon> <maxlat> --out out/subset
   ```

4. **Merge**:

   ```bash
   python scripts/merge_esurvey_cropscan.py --cropscan data/cache/cropscan_utm.parquet \
       --esurvey data/cache/esurvey_utm.parquet \
       --cropscan-original data/extracted/cropscan/<cropscan>.shp \
       --out out/v1/result.gpkg
   ```

   Writes `result.gpkg` (EPSG:4326, for delivery and QGIS) and `result.parquet`
   (UTM, for checks). Write every run to a new folder so a file open in QGIS is
   never overwritten. On a subset, pass the subset files and the same
   `--cropscan-original`.

5. **QA on the satellite basemap** (optional):

   ```bash
   python scripts/cache_tiles.py --layer data/cache/esurvey_utm.parquet --out data/tiles --zoom 18
   python scripts/render_views.py --views views.csv --out qa/views \
       --layer cropscan_in=data/cache/cropscan_utm.parquet:yellow \
       --layer esurvey_in=data/cache/esurvey_utm.parquet:red \
       --layer "out_clipped=out/v1/result.parquet?source=esurvey_clipped:magenta"
   ```

   `views.csv` has the columns `name,lat,lon,half_m`. Tiles are cached on disk,
   so each tile is downloaded once.

## Tuning

The thresholds are constants at the top of `scripts/merge_esurvey_cropscan.py`
(gap closing, 5 m edge tolerance, minimum empty field, small-piece classes).
Change one, run on a subset into a new folder, and compare in QGIS.
