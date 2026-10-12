# How the reconciliation works

This document explains, step by step, what the pipeline does, **why** each step
exists and **how** it is implemented. It is written for someone who is new to
the project. Read it once from top to bottom; afterwards the code in
`scripts/merge_esurvey_cropscan.py` should be easy to follow.

---

## 1. The problem

We have two maps of the same crop (sugarcane) over the same land.

| | **eSurvey** (survey polygons) | **CropScan** (model polygons) |
|---|---|---|
| Where it comes from | a field survey done on the ground | a satellite-based crop classification model |
| Strong point | clean, straight field boundaries that the client trusts | knows where the crop really is at scan time |
| Weak point | neighbouring fields are often merged into one big block; some fields inside a block have no crop | edges are jagged (they follow image pixels), stop a few metres short of field edges, and break at roads and bunds |

The client wants **one** layer:

* the boundaries should look like the eSurvey (straight, field-like), but
* a field should only be there if CropScan shows the crop in it.

So the job is: *keep eSurvey geometry, but cut away every field that CropScan
does not confirm, and add CropScan fields that the survey missed.*

---

## 2. The rules (what the client asked for)

1. **eSurvey with CropScan**: keep the eSurvey polygon, but cut out every field
   that has no CropScan in it, whether that field is on the side of the block or
   in the middle, small or big. Cuts must be straight lines that look like field
   boundaries, not the jagged CropScan edge.
2. **Misalignment is not a reason to cut**: CropScan often stops a few metres
   inside the real field edge, and there are thin gaps along roads and bunds
   (the raised earth lines between fields). These must not cause cuts.
3. **eSurvey without CropScan**: drop it. CropScan is the proof that the crop is there.
4. **CropScan without eSurvey**: keep it **exactly** as delivered, with not a single
   vertex changed.
5. **CropScan reaching outside eSurvey**: keep the outside part (it is a field the
   survey missed), except thin strips that only come from misalignment along the
   eSurvey edge.
6. **Review help**: every polygon gets a `category`; small leftover pieces and
   pieces of very large CropScan polygons get their own categories, so a
   reviewer can find them quickly. Areas are in **acres with 3 decimals**.

---

## 3. The pipeline at a glance

```
 inputs (S3 / GCS)            download_inputs.py  /  aws s3 sync
        │
        ▼
 prepare_inputs.py            one AOI, metric CRS, repaired polygons  ──►  data/cache/<aoi>/*.parquet
        │
        ▼
 merge_esurvey_cropscan.py    the rules above, in parallel, resumable  ──►  out/<aoi>/<version>/result.gpkg
        │
        ├──► upload_outputs.py         final files to GCS, verified by size
        └──► render_views.py           QA pictures on a satellite basemap (cache_tiles.py caches the tiles)
```

`make_subset.py` cuts a small test area out of the prepared data, so rule
changes can be tried in seconds instead of minutes.

---

## 4. Step by step

### Step 1 – Get the inputs

* `download_inputs.py` (GCS) or `aws s3 sync` (S3) copy the shapefiles to
  `data/`. Both skip files that are already there with the same size, so they
  can be run again after a stop.

### Step 2 – Prepare (`prepare_inputs.py`)

**What it does**

1. Reads CropScan and eSurvey. With `--aoi`, it reads only what overlaps the AOI.
2. Keeps a polygon in the AOI when an **interior point** of the polygon lies in
   the AOI (`point_on_surface`). Polygons are **never cut** at the AOI line.
3. Projects everything to the local **UTM** zone (a metric coordinate system).
4. eSurvey only: splits multipolygons into single polygons and repairs invalid
   rings (`make_valid`). Its own `Id` field is kept as `esurvey_Id`.
5. Writes `cropscan_utm.parquet` and `esurvey_utm.parquet`.

**Why**

* *Interior point, not cutting*: rule 4 says CropScan must stay exactly as
  delivered. Cutting at the AOI line would change polygons. With the interior
  point every polygon belongs to exactly one AOI, so two AOIs never count the
  same polygon twice.
* *UTM*: all distances in the rules are in metres (5 m, 12 m...) and areas in
  acres. In longitude/latitude degrees those numbers would be wrong.
* *Repair*: the survey file has self-intersecting rings; geometry operations
  fail or give wrong areas on invalid polygons.
* *Row id kept*: the CropScan index is the feature id of the original file. It
  becomes `cs_id` in the output and is used at the end to copy the original
  geometry back (Step 3e).
* *GeoParquet*: much faster to read than a shapefile and keeps the CRS.

### Step 3 – Merge (`merge_esurvey_cropscan.py`)

The merge has two parts: one job per **eSurvey polygon** (3a–3c) and one job per
**CropScan polygon** (3d). Both run in parallel (Section 5).

#### 3a. Look only at the neighbourhood of the eSurvey polygon

For one eSurvey polygon `g`:

* take a window = `g` grown by **30 m** (`WINDOW_M`);
* find the CropScan polygons that touch the window with a spatial index
  (`STRtree`) and clip them to the window.

*Why a window*: the decision for `g` only depends on CropScan close to it. A
small window keeps every operation small and fast, and lets many polygons be
processed at the same time. 30 m is more than twice the gap-closing distance
below, so the window edge never changes the result inside `g`.

#### 3b. Clean CropScan and bridge roads and bunds

1. **Fill tiny holes** in CropScan smaller than **200 m²** (`HOLE_FILL_HA`). These
   are noise pixels, not real empty fields.
2. **Closing** with **6 m** (`GAP_CLOSE_M`): grow CropScan by 6 m, then shrink it
   by 6 m. This fills gaps up to about **12 m** wide, such as a road or bund
   between two cane fields, while the outer shape stays where it was.
3. In big eSurvey blocks (≥ **5 ha**, `BIG_HA`), lone CropScan specks smaller
   than **200 m²** (`ISLAND_HA`) are ignored, so a single noisy pixel cannot keep
   a whole empty field. This second closing is only done when such specks exist.
4. `es_cover` = share of `g` covered by the closed CropScan (written to the output).
   If it is 0, the eSurvey polygon is dropped (rule 3).

#### 3c. Find empty fields and cut straight (`straight_cut`)

This is the heart of the method. The question is: *which parts of the eSurvey
polygon `g` are empty fields, and where exactly do we cut?*

1. **Find the field direction** (`dominant_angle`). Fields are laid out on a grid
   that is usually a bit rotated. The function measures the direction of every
   edge of `g`, weighted by edge length, and takes the most common direction
   (0–90°).
2. **Rotate** `g` and the local CropScan by minus that angle, so field edges
   become horizontal and vertical.
3. **Rasterise** both on a **2.5 m** pixel grid (`PIX`) and group pixels into
   **5 m cells** (`CELL` = 2 pixels). A cell is "CropScan" when at least **50 %**
   of its inside-`g` area is CropScan (`CELL_KEEP`).
4. **Tolerances** (rule 2):
   * a *closing* by one cell bridges gaps of up to ~10 m between CropScan cells;
   * along the eSurvey **outline**, a cell is kept when CropScan is within one
     cell (**5 m**, `TOL_CELLS`). This keeps the survey edge where CropScan only
     stops a little short of it.
5. Empty areas smaller than **0.01 ha** (`NOISE_HA`) are not cut (too small to
   be a field).
6. Kept parts only one cell (5 m) wide are removed (no slivers).
7. The kept cells are turned back into polygons, rotated back, and intersected
   with `g`.

*Why this gives straight cuts*: the cells follow the field direction, so the
cut lines are straight and parallel to the field edges, and they meet the
original eSurvey outline cleanly. Where nothing is cut, the original eSurvey
polygon is kept (`esurvey_full`); otherwise the cut result is kept
(`esurvey_clipped`).

*Why the outline tolerance applies only to the outline*: an earlier version
grew CropScan everywhere by 5–8 m before cutting. Two CropScan fields with an
empty 20 m field between them then "closed" over the empty field, and whole
empty fields stayed in the output. Applying the tolerance only on the outline
fixed that.

Each output part also gets `cs_id`: the CropScan polygon it overlaps most.

#### 3d. CropScan outside eSurvey (one job per CropScan polygon)

For one CropScan polygon `c`:

* **No eSurvey touches it** → it is kept as is (`cropscan_original`, rule 4).
* **Otherwise** the part of `c` outside all eSurvey polygons is computed. A
  square-cornered *opening* of **4 m** (`SPILL_OPEN_M`) removes strips narrower
  than about **8 m**; these are only misalignment along the eSurvey edge. Pieces
  narrower than **8 m** overall or smaller than **0.01 ha** are dropped. The rest
  is kept (`cropscan_clipped`, rule 5).

*Why square corners*: a normal (round) opening would round off field corners.
The mitre join keeps corners as CropScan drew them.

#### 3e. Attributes, original geometry and writing

* `area_ac` = area in acres (1 acre = 4046.8564224 m²), rounded to 3 decimals,
  computed in UTM.
* `category` (see Section 6).
* Pieces smaller than **0.01 ha** that were *created* by the cuts are removed;
  untouched CropScan is never filtered.
* **Exact original geometry**: for every untouched CropScan polygon, the
  geometry is read again from the **original file** by its feature id and
  written as it is. Projecting to UTM and back would move every vertex by tiny
  amounts; copying from the original guarantees *not even the vertex count
  changes*.
* Output: `result.parquet`, a GeoParquet in EPSG:4326 (fast to write and read;
  QGIS opens it directly). `--gpkg` also writes a GeoPackage.

### Step 4 – Upload (`upload_outputs.py`)

Uploads the final files to GCS in parallel chunks, then reads the size back from
the bucket and compares it with the local file. Files already in the bucket
with the same size are skipped.

### Step 5 – QA on a satellite basemap

* `cache_tiles.py` downloads the satellite tiles once (zoom 18) and keeps them on
  disk; it skips tiles that are already there.
* `render_views.py` draws, for a list of places, the inputs on the left and the
  output (filled by source) on the right.

The rules were tuned by looking at many such pictures **case by case** on a
test area, from the client's point of view. Keep in mind that the basemap date
differs from the crop season: a bare field on the basemap does not prove there
is no crop. Use the basemap mainly for shapes and field boundaries.

---

## 5. Speed and resumability

### Parallel windows

* Every eSurvey polygon and every CropScan polygon is an independent job on its
  own small window (Section 3a). Jobs run in a pool of processes, one per CPU
  core (`--workers` to limit it).
* The inputs are loaded once; worker processes are **forked**, so they share the
  inputs without copying them.
* Jobs are handed out **biggest first, one polygon at a time**. Otherwise one
  worker could end up with all the slow, big polygons while the others wait.
* `cs_id` is computed inside the jobs instead of a single-core spatial join at
  the end.
* The exact original CropScan geometry is stored once as GeoParquet
  (`cropscan_original.parquet`) by `prepare_inputs.py`; reading it back by id
  takes seconds (doing the same from the shapefile took many minutes).

Profiling showed where time went: in the first parallel version, half of the
run time was a single-core step and the work was badly balanced. Fixing that
took the reference run from 489 s to 177 s with exactly the same output.
**Always profile before optimising**, and compare the output after every
speed-up (same categories, ids and areas).

### Resumable (spot machines)

The machine may be stopped at any time (spot instances are taken back). So:

* **Saved groups**: the jobs are split into groups (`ES_GROUP` eSurvey polygons,
  `CS_GROUP` CropScan polygons). Each group mixes big and small polygons, so
  groups take about the same time. When a group finishes, its result is saved in
  `result_parts/`. Running the same command again loads finished groups and only
  computes the rest.
* **Safe reuse**: `result_parts/KEY` is a fingerprint of the input files and the
  merge script. If either changes, the saved groups are discarded automatically,
  so old results are never mixed with new rules.
* **Atomic writes**: every file is written under a temporary name and then
  renamed. A stop during writing leaves the old file or no file, never a broken
  one.
* Prepare, download and upload all skip work that is already done.

### Robustness

Very rarely, the geometry engine (GEOS) fails on two almost identical edges
("non-noded intersection"). The job then retries that one polygon with
coordinates snapped to a **1 mm** grid instead of stopping the whole run.

---

## 6. Output fields and categories

| Field | Meaning |
|---|---|
| `fid` | running number |
| `predicted` | class code, 1 = Sugarcane |
| `source` | technical origin: `esurvey`, `esurvey_clipped`, `cropscan`, `cropscan_spill` |
| `category` | class for styling and review (below) |
| `es_cover` | share of the eSurvey polygon covered by (closed) CropScan |
| `es_id` | row of the eSurvey polygon in the prepared file, -1 if none |
| `esurvey_Id` | the eSurvey's own `Id` field, when the input has one |
| `cs_id` | feature id of the CropScan polygon in the original file, -1 if none |
| `area_ac` | area in acres, 3 decimals |

| `category` | Meaning |
|---|---|
| `esurvey_full` | eSurvey polygon kept whole: CropScan covers it, apart from roads, bunds and edge misalignment |
| `esurvey_clipped` | eSurvey polygon with empty fields cut out |
| `cropscan_original` | CropScan with no eSurvey around it, original geometry |
| `cropscan_clipped` | part of a CropScan polygon outside eSurvey, 0.20 acres or more |
| `cropscan_small_lt_0.10ac` | same, smaller than 0.10 acres (review) |
| `cropscan_small_0.10_0.15ac` | same, 0.10 to 0.15 acres (review) |
| `cropscan_small_0.15_0.20ac` | same, 0.15 to 0.20 acres (review) |
| `cropscan_large_review` | CropScan outside eSurvey coming from a CropScan polygon larger than 250 acres; such huge polygons often cover bare land, so a person should look at them |

---

## 7. Tuning cheat sheet

All thresholds are constants at the top of `scripts/merge_esurvey_cropscan.py`.

| Constant | Value | Effect when larger |
|---|---|---|
| `GAP_CLOSE_M` | 6 m | wider roads / gaps are bridged (risk: empty narrow fields kept) |
| `HOLE_FILL_HA` | 0.02 ha | bigger holes in CropScan are treated as noise |
| `BIG_HA` | 5 ha | fewer eSurvey polygons count as "big blocks" (speck filter) |
| `ISLAND_HA` | 0.02 ha | bigger lone CropScan bits are ignored in big blocks |
| `PIX`, `CELL` | 2.5 m, 2 | coarser cut grid: faster, cuts less exact (cell = 5 m) |
| `CELL_KEEP` | 0.5 | a cell needs more CropScan to count as crop |
| `TOL_CELLS` | 1 (5 m) | eSurvey edge kept further from CropScan |
| `NOISE_HA` | 0.01 ha | bigger empty spots are not cut |
| `SPILL_OPEN_M`, `SPILL_MIN_WIDTH_M` | 4 m, 8 m | wider strips outside eSurvey are removed |
| `SMALL_CLASSES` | 0.10 / 0.15 / 0.20 ac | size classes for review |
| `LARGE_CS_AC` | 250 ac | fewer CropScan polygons flagged as large |

How to change a rule safely:

1. make a test area with `make_subset.py`;
2. change one constant;
3. run the merge into a **new** output folder (never overwrite a file open in QGIS);
4. compare with the previous version in QGIS and on the basemap;
5. only then run the full AOI.
