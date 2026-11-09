"""Star-level train/val/test splitting (all TCEs of a star land in the same split)"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

KEY_COL = "kepid"
LABEL_COL = "label"
SPLIT_NAMES = ("train", "val", "test")


def make_star_table(
    labels: pd.DataFrame, star_col: str = KEY_COL, label_col: str = LABEL_COL
) -> pd.DataFrame:
    """One row per star: [star_col, 'star_label'], 1 if any of its TCEs is a planet."""
    collapsed_df = labels.groupby(star_col).agg(star_label=(label_col, "max")).reset_index()

    assert len(collapsed_df) == labels[star_col].nunique(), "expected one row per star"
    return collapsed_df


def make_star_splits(
    labels: pd.DataFrame,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    star_col: str = KEY_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Return a table [star_col, 'split'], one row per star, sorted by star."""
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"train + val + test: {train + val + test} != 1")

    stars = make_star_table(labels, star_col, label_col)

    train_stars, rest = train_test_split(
        stars, test_size=val + test, stratify=stars["star_label"], random_state=seed
    )

    val_stars, test_stars = train_test_split(
        rest, test_size=test / (val + test), stratify=rest["star_label"], random_state=seed
    )

    out = pd.concat(
        [
            train_stars.assign(split="train"),
            val_stars.assign(split="val"),
            test_stars.assign(split="test"),
        ]
    )

    return out[[star_col, "split"]].sort_values(star_col).reset_index(drop=True)


def verify_no_leakage(labels: pd.DataFrame, splits: pd.DataFrame, star_col: str = KEY_COL) -> None:
    """Raise ValueError if any star is missing, duplicated, or in more than one split."""
    if not splits[star_col].is_unique:
        raise ValueError("splits['star_col'] contains a star appearing more than once")

    missing = set(labels[star_col]) - set(splits[star_col])
    if missing:
        raise ValueError(f"{len(missing)} stars in labels have no splits")

    extras = set(splits[star_col]) - set(labels[star_col])
    if extras:
        raise ValueError(f"{len(extras)} stars in splits are not in labels")

    unknown = set(splits["split"]) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unexpected split names: {sorted(unknown)}")


def summarize(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    star_col: str = KEY_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Per split: number of stars, TCEs, planets, and planet rate"""
    merged = labels.merge(splits, on=star_col)
    rows = []
    for name in SPLIT_NAMES:
        m = merged[merged["split"] == name]
        rows.append(
            {
                "split": name,
                "stars": m[star_col].nunique(),
                "tces": len(m),
                "planet": int(m[label_col].sum()),
                "planet_rate": round(float(m[label_col].mean()), 4),
            }
        )
    return pd.DataFrame(rows)
