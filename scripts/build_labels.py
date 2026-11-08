"""Build data/labels.parquet from kepler DR24 TCE table

Usage:
    python scripts/build_labels.py           #use cached table if parent
    python scripts/build_labels.py --refresh #re-download from the archive
"""

from __future__ import annotations

import argparse

import pandas as pd

from exotransit.datasets.fetch_tce import fetch_tce_table
from exotransit.datasets.labels import make_labels
from exotransit.utils import load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    cfg = load_config()
    cache_path = resolve(cfg["paths"]["tce_raw"])
    labels_path = resolve(cfg["paths"]["labels"])
    unk_path = resolve(cfg["paths"]["tces_unk"])

    raw, source = fetch_tce_table(cache_path, refresh=args.refresh)
    print(f"{source}: {len(raw)} rows")

    labeled, unk = make_labels(raw)

    if len(labeled) + len(unk) != len(raw):
        raise ValueError(
            f"row count mismatch: {len(labeled)} labeled + {len(unk)} unk != {len(raw)} raw"
        )
    both = pd.concat([labeled, unk])
    if both.duplicated(subset=["kepid", "tce_plnt_num"]).any():
        n_dup = both.duplicated(subset=["kepid", "tcet_plnt_num"]).sum()
        raise ValueError(f"{n_dup} TCEs appear in both the labeled and unk tables")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(labels_path, index=False)
    unk.to_parquet(unk_path, index=False)

    print(f"labeled: {len(labeled)} rows")
    print(labeled["disposition"].value_counts().to_string())
    print(labeled.groupby("disposition")["fit_failed"].sum().to_string())
    print(f"planet rate:m {labeled['label'].mean():.3f}")
    print(f"unk: {len(unk)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
