"""
1. Turn the raw DR24 TCE table into a labeled table and an UNK table
2. Run checks - Nothing is zero or negative, The transit is shorter than the orbit,
No row is repeated.
"""

from __future__ import annotations

import pandas as pd

LABEL_MAP = {"PC": 1, "AFP": 0, "NTP": 0}
EXPECTED = {"PC", "AFP", "NTP", "UNK"}

RENAME = {
    "tce_period": "period_day",
    "tce_time0bk": "epoch_bkjd",
    "tce_depth": "depth_ppm",
    "tce_model_snr": "snr",
}

FINAL_COLS = [
    "kepid",
    "tce_plnt_num",
    "disposition",
    "label",
    "period_day",
    "epoch_bkjd",
    "duration_day",
    "depth_ppm",
    "snr",
    "tce_quarters",
    "tce_num_transits",
]


def make_labels(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labeled, unk). 'raw' is the table from fetch_tce_table"""
    df = raw.copy()

    found = set(df["av_training_set"].unique())  # get list of distinct values

    # check any unexpected labels
    assert found <= EXPECTED, f"unexpected dispositions: {found - EXPECTED}"

    df["disposition"] = df["av_training_set"]

    df["duration_day"] = df["tce_duration"] / 24

    df = df.rename(columns=RENAME)

    is_unk = (
        df["disposition"] == "UNK"
    )  # turn cols into True and False, True for the 4,630 UNK rows
    unk = df[is_unk].copy()  # turn those true rows and becomes "unk" table
    labeled = df[~is_unk].copy()  # [~is_unk] flips the ture/ false, and keeps the non-unk rows

    labeled["label"] = (
        labeled["disposition"].map(LABEL_MAP).astype("int8")
    )  # turn label into 1 or 0.

    labeled = labeled.sort_values(["kepid", "tce_plnt_num"]).reset_index(drop=True)
    unk = unk.sort_values(["kepid", "tce_plnt_num"]).reset_index(drop=True)

    labeled = labeled[FINAL_COLS]
    unk = unk[[c for c in FINAL_COLS if c != "label"]]

    validate(labeled)
    validate(unk)
    return labeled, unk


def validate(df: pd.DataFrame) -> None:
    """Sanity checks. Raise AssertionError with a clear message if any fail."""
    for col in ["period_day", "duration_day", "depth_ppm"]:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            raise ValueError(f"{bad} rows have {col} <= 0")

    bad = (df["duration_day"] >= df["period_day"]).sum()
    if bad > 0:
        raise ValueError(f"{bad} rows have duration_day >= period_day")

    bad = df.duplicated(subset=["kepid", "tce_plnt_num"]).sum()
    if bad > 0:
        raise ValueError(f"{bad} have duplicated (kepid, tce_plnt_num rows")
