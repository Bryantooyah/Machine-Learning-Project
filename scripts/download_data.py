"""Download the HDB resale flat price dataset from data.gov.sg.

Dataset: "Resale flat prices based on registration date from Jan-2017 onwards"
Dataset ID: d_8b84c4ee58e3cfc0ece0d773c8ca6abc

The CSV is already committed under data/, so you only need to rerun this to
refresh it (the source is updated as new transactions are registered).

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "resale_flat_prices_2017_onwards.csv"


def fetch_download_url() -> str:
    """Ask the data.gov.sg API for a temporary signed S3 URL for the CSV."""
    import json

    with urllib.request.urlopen(POLL_URL, timeout=60) as response:
        payload = json.load(response)

    if payload.get("code") != 0:
        raise RuntimeError(f"API error: {payload.get('errorMsg')!r}")

    data = payload.get("data", {})
    if data.get("status") != "DOWNLOAD_SUCCESS":
        raise RuntimeError(f"Download not ready: {data.get('status')!r}")

    return data["url"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Requesting download URL from data.gov.sg ...")
    url = fetch_download_url()

    print(f"Downloading to {args.out} ...")
    urllib.request.urlretrieve(url, args.out)

    size_mb = args.out.stat().st_size / 1024**2
    print(f"Done. {args.out.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
