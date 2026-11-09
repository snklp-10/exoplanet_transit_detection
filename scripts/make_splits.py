"""Create the frozen star-level train/val/test split from data/labels.parquet

Writes data/splits.csv The split is created once and committed. the script refuses
to overwrite it unless --force os given.
"""

from __future__ import annotations

import argparse

import pandas as pd

from exotransit.datasets.splits import make_star_splits, summarize, verify_no_leakage
from exotransit.utils import load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config yaml")
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels_path = resolve(cfg["paths"]["labels"])
    splits_path = resolve(cfg["paths"]["splits"])

    if splits_path.exists() and not args.force:
        print(f"{splits_path} already exists and is frozen. Use --force to overwrite")
        return 1

    labels = pd.read_parquet(labels_path)
    splits = make_star_splits(
        labels,
        train=cfg["split"]["train"],
        val=cfg["split"]["val"],
        test=cfg["split"]["test"],
        seed=cfg["seed"],
    )
    verify_no_leakage(labels, splits)

    splits.to_csv(splits_path, index=False)
    print(f"Wrote {splits_path}")
    print(summarize(labels, splits).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
