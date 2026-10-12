"""Upload final outputs to Google Cloud Storage and verify every file by size.

Files already in the bucket with the same size are skipped, so the upload can be
run again after a stop. Large files go up in parallel chunks.

Usage:
    python scripts/upload_outputs.py --key <service_account.json> --bucket <bucket> \
        --prefix <prefix/> out/aoi_p1/v2/result.gpkg out/aoi_p1/v2/result.parquet
"""
import argparse
import os

from google.cloud import storage
from google.cloud.storage import transfer_manager

CHUNK = 32 * 1024 * 1024  # 32 MB per chunk
WORKERS = 32  # parallel network streams


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--key", required=True, help="service account JSON")
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True, help="folder in the bucket, ending with /")
    a = p.parse_args()

    bucket = storage.Client.from_service_account_json(a.key).bucket(a.bucket)
    bad = 0
    for path in a.files:
        name = a.prefix + os.path.basename(path)
        size = os.path.getsize(path)
        blob = bucket.get_blob(name)
        if blob is not None and blob.size == size:
            print(f"skip (already there)  {name}")
            continue
        blob = bucket.blob(name, chunk_size=CHUNK)
        if size > CHUNK:
            transfer_manager.upload_chunks_concurrently(path, blob, chunk_size=CHUNK, max_workers=WORKERS)
        else:
            blob.upload_from_filename(path)
        blob = bucket.get_blob(name)
        ok = blob is not None and blob.size == size
        bad += not ok
        print(f"{'ok' if ok else 'SIZE MISMATCH'}  {name}  {size / 1e6:.1f} MB")
    if bad:
        raise SystemExit(f"{bad} file(s) failed the size check")


if __name__ == "__main__":
    main()
