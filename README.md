# Exoplanet Transit Detection with Deep Learning

Classify periodic dips in Kepler light curves as real planet transits or false positives
(e.g. eclipsing binaries), and measure honestly where a 1D CNN improves on a classical BLS baseline.

> Status: week 1 of 6, data and labels. Results sections are filled in as they are produced.

## Progress

- [x] Environment, repo skeleton, config
- [x] Kepler DR24 TCE table fetched through the NASA Exoplanet Archive TAP service, with a local cache
- [x] Label table: 15,737 labeled TCEs (PC 3,600, AFP 9,596, NTP 2,541) plus 4,630 unlabeled (UNK) TCEs kept separately
- [x] Validation checks and unit tests for the labeling step
- [x] Frozen star-level train/val/test split
- [ ] Light-curve download and cache
- [ ] Data manifest and EDA figures
- [ ] Preprocessing, BLS baseline, 1D CNN (weeks 2 to 6)

## Build the label table

```bash
python scripts/build_labels.py             # uses the cached table if present
python scripts/build_labels.py --refresh   # re-download from the archive
```

Writes `data/labels.parquet` (committed) and `data/tces_unk.parquet` (git-ignored).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11 or 3.12 recommended
pip install -e ".[dev,boost,detrend,track]"            # or: make install
python scripts/check_env.py --network                   # verifies imports, BLS, torch, archive access
```

For GPU training, install the CUDA build of PyTorch first from https://pytorch.org, then run the line above.

## Layout

```
configs/            config.yaml (pipeline parameters) and experiments/ (per-ablation configs)
data/               raw/, processed/ (git-ignored); labels.parquet and splits.csv are committed
notebooks/          01_eda.ipynb
scripts/            command-line entry points (one per pipeline stage)
src/exotransit/     datasets/, preprocessing/, baselines/, models/, evaluation/, utils.py
tests/              one test file per module
runs/, checkpoints/ training logs and model weights (git-ignored)
reports/            figures/, results/ (metric tables), writeup.md
```

## Evaluation protocol (fixed in week 1)

- Split by star (`kepid`), never by TCE; frozen in `data/splits.csv`.
- Headline metrics: PR-AUC, and recall at fixed precision, on held-out test stars.
- Normalization and thresholds are fit on train only.

## Results

_To be added._
