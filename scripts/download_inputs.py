"""Download the raw input files from a GCS prefix into a local folder.

Large files are fetched in parallel chunks; files that already exist locally
with the same size are skipped.

Usage:
    python scripts/download_inputs.py --bucket <bucket> --prefix <prefix/> \
        --key <service_account.json> --out data/raw
"""
import argparse
import os

from google.cloud import storage
from google.cloud.storage import transfer_manager

CHUNK = 32 * 1024 * 1024  # 32 MB per chunk
WORKERS = 32  # parallel network streams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--key", required=True, help="service account JSON")
    p.add_argument("--out", default="data/raw")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    client = storage.Client.from_service_account_json(a.key)
    blobs = [b for b in client.list_blobs(a.bucket, prefix=a.prefix) if not b.name.endswith("/")]

    for b in blobs:
        dst = os.path.join(a.out, os.path.basename(b.name))
        if os.path.exists(dst) and os.path.getsize(dst) == b.size:
            print(f"skip (exists)  {dst}")
            continue
        transfer_manager.download_chunks_concurrently(b, dst, chunk_size=CHUNK, max_workers=WORKERS)
        ok = os.path.getsize(dst) == b.size
        print(f"{'ok' if ok else 'SIZE MISMATCH'}  {dst}  {b.size / 1e6:.1f} MB")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
