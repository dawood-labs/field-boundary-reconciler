# field-boundary-reconciler

Turns model-detected crop polygons (CropScan) into polygons that follow surveyed
field boundaries (eSurvey). Where both exist, the output keeps the surveyed
boundary but cuts away every field that has no detected crop. Where only the
model found the crop, the model polygon is kept exactly as it is.

Detailed explanation of every step (what, why and how): [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Rules

| Case | Output | `category` |
|---|---|---|
| eSurvey polygon fully covered by CropScan (thin road / bund gaps ignored) | eSurvey polygon as it is | `esurvey_full` |
| eSurvey polygon partly covered | eSurvey cut with straight lines along the field directions; every empty field (inside or on the side, small or big) is removed. The outline stays where CropScan is within 5 m of it. | `esurvey_clipped` |
| eSurvey polygon with no CropScan | dropped | – |
| CropScan polygon with no eSurvey | original CropScan geometry, unchanged (same vertices) | `cropscan_original` |
| Part of a CropScan polygon outside eSurvey | kept, except edge misalignment strips narrower than ~8 m | `cropscan_clipped` |
| Same, but smaller than 0.20 acres | kept, flagged by size for review | `cropscan_small_lt_0.10ac`, `cropscan_small_0.10_0.15ac`, `cropscan_small_0.15_0.20ac` |
| CropScan outside eSurvey that comes from a CropScan polygon larger than 250 acres | kept, flagged for review | `cropscan_large_review` |

Output fields: `fid`, `predicted` (1 = Sugarcane), `source`, `category`,
`es_cover` (share of the eSurvey polygon covered by CropScan), `es_id`,
`esurvey_Id` (the eSurvey's own `Id` field, when the input has one),
`cs_id` (row number of the source polygons, -1 if none), `area_ac`
(acres, 3 decimals).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, any machine. The merge uses every core (`--workers` to limit it).
Each eSurvey and CropScan polygon is processed in its own small window, so time
and memory grow linearly with the area. Reference run, 8 cores: ~24,000
CropScan polygons (27 million vertices) and ~10,000 eSurvey polygons in about
3 minutes, peak memory 1.7 GB.

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

   Inputs on S3 instead: `aws s3 sync s3://<bucket>/<prefix>/ data/s3_input/` (only missing or
   changed files are fetched, so it can simply be run again).

2. **Prepare the inputs** (UTM GeoParquet, eSurvey split into single polygons and repaired),
   optionally for one AOI (polygons inside it by an interior point; nothing is cut at the AOI line):

   ```bash
   python scripts/prepare_inputs.py --cropscan <cropscan.shp> --esurvey <esurvey.shp> \
       --aoi <aoi.shp> --out data/cache/<aoi_name>
   ```

3. **Optional: make a small test area** to try rule changes quickly:

   ```bash
   python scripts/make_subset.py --cropscan data/cache/cropscan_utm.parquet \
       --esurvey data/cache/esurvey_utm.parquet --aoi <minlon> <minlat> <maxlon> <maxlat> --out out/subset
   ```

4. **Merge**:

   ```bash
   python scripts/merge_esurvey_cropscan.py --cropscan data/cache/<aoi_name>/cropscan_utm.parquet \
       --esurvey data/cache/<aoi_name>/esurvey_utm.parquet \
       --cropscan-original data/cache/<aoi_name>/cropscan_original.parquet \
       --out out/<aoi_name>/v1/result.parquet
   ```

   Writes `result.parquet`, a GeoParquet in EPSG:4326 (fast; QGIS opens it
   directly); add `--gpkg` for a GeoPackage as well. Write every run to a new
   folder so a file open in QGIS is never overwritten.

   **Resumable:** results are saved in small groups under `result_parts/`. If the
   machine stops (e.g. a reclaimed spot instance), run the same command again: it
   reuses the finished groups and only computes the rest. The saved groups are
   discarded automatically when the inputs or the script change. Final files are
   written under a temporary name and renamed, so a stop never leaves a broken file.

5. **Upload the final outputs** to Google Cloud Storage (skips files already there
   with the same size; verifies every file by size):

   ```bash
   python scripts/upload_outputs.py --key <service_account.json> --bucket <bucket> \
       --prefix <prefix/> out/<aoi_name>/v1/result.parquet
   ```

6. **QA on the satellite basemap** (optional):

   ```bash
   python scripts/cache_tiles.py --layer data/cache/<aoi_name>/esurvey_utm.parquet --out data/tiles --zoom 18
   python scripts/render_views.py --views views.csv --out qa/views \
       --layer cropscan_in=data/cache/<aoi_name>/cropscan_utm.parquet:yellow \
       --layer esurvey_in=data/cache/<aoi_name>/esurvey_utm.parquet:red \
       --layer "out_clipped=out/<aoi_name>/v1/result.parquet?source=esurvey_clipped:magenta"
   ```

   `views.csv` has the columns `name,lat,lon,half_m`. Tiles are cached on disk,
   so each tile is downloaded once.

## Tuning

The thresholds are constants at the top of `scripts/merge_esurvey_cropscan.py`
(gap closing, 5 m edge tolerance, minimum empty field, small-piece classes).
Change one, run on a subset into a new folder, and compare in QGIS.
