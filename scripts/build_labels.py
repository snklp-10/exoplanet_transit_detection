"""Build data/labels.parquet from kepler DR24 TCE table

Usage:
    python scripts/build_labels.py           #use cached table if parent
    python scripts/build_labels.py --refresh #re-download from the archive
"""

from __future__ import annotations

import argparse

from exotransit.datasets.fetch_tce import fetch_tce_table
from exotransit.utils import load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    cfg = load_config()
    cache_path = resolve(cfg["paths"]["tce_raw"])

    df, source = fetch_tce_table(cache_path, refresh=args.refresh)
    print(f"{source}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
