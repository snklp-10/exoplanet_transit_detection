import numpy as np
import pandas as pd
import pytest

from exotransit.datasets.splits import (
    SPLIT_NAMES,
    make_star_splits,
    make_star_table,
    summarize,
    verify_no_leakage,
)


def _synthetic_labels(n_stars=1000, planet_star_rate=0.15, seed=0):
    """Stars with 1-3 TCEs each; some planet stars also have non-planet TCEs."""
    rng = np.random.default_rng(seed)
    rows = []
    for kepid in range(1, n_stars + 1):
        is_planet_star = rng.random() < planet_star_rate
        for tce in range(rng.integers(1, 4)):
            label = int(is_planet_star and (tce == 0 or rng.random() < 0.5))
            rows.append({"kepid": kepid, "tce_plnt_num": tce + 1, "label": label})
    return pd.DataFrame(rows)


def test_star_table_is_one_row_per_star_with_max_label():
    labels = pd.DataFrame({"kepid": [1, 1, 2, 3, 3], "label": [0, 1, 0, 0, 0]})
    table = make_star_table(labels)
    assert len(table) == 3
    assert table.set_index("kepid")["star_label"].to_dict() == {1: 1, 2: 0, 3: 0}


def test_no_star_in_two_splits_and_all_stars_covered():
    labels = _synthetic_labels()
    splits = make_star_splits(labels, seed=42)
    verify_no_leakage(labels, splits)
    merged = labels.merge(splits, on="kepid")
    assert (merged.groupby("kepid")["split"].nunique() == 1).all()
    assert len(splits) == labels["kepid"].nunique()


def test_fractions_are_close_to_requested():
    labels = _synthetic_labels()
    splits = make_star_splits(labels, train=0.8, val=0.1, test=0.1, seed=42)
    frac = splits["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.01)
    assert frac["val"] == pytest.approx(0.1, abs=0.01)
    assert frac["test"] == pytest.approx(0.1, abs=0.01)


def test_is_deterministic_and_seed_sensitive():
    labels = _synthetic_labels()
    a = make_star_splits(labels, seed=42)
    b = make_star_splits(labels, seed=42)
    c = make_star_splits(labels, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_star_level_planet_rate_is_preserved_across_splits():
    labels = _synthetic_labels(n_stars=3000)
    splits = make_star_splits(labels, seed=42)
    stars = make_star_table(labels).merge(splits, on="kepid")
    overall = stars["star_label"].mean()
    per_split = stars.groupby("split")["star_label"].mean()
    for name in SPLIT_NAMES:
        assert per_split[name] == pytest.approx(overall, abs=0.01)


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        make_star_splits(_synthetic_labels(), train=0.7, val=0.1, test=0.1)


def test_verify_detects_missing_star():
    labels = _synthetic_labels()
    splits = make_star_splits(labels).iloc[1:]
    with pytest.raises(ValueError):
        verify_no_leakage(labels, splits)


def test_verify_detects_extra_star():
    labels = _synthetic_labels()
    splits = make_star_splits(labels)
    extra = pd.DataFrame({"kepid": [999999], "split": ["train"]})
    with pytest.raises(ValueError):
        verify_no_leakage(labels, pd.concat([splits, extra], ignore_index=True))


def test_verify_detects_star_in_two_splits():
    labels = _synthetic_labels()
    splits = make_star_splits(labels)
    first = splits.iloc[[0]].assign(split="test")
    with pytest.raises(ValueError):
        verify_no_leakage(labels, pd.concat([splits, first], ignore_index=True))


def test_verify_detects_unknown_split_name():
    labels = _synthetic_labels()
    splits = make_star_splits(labels)
    splits.loc[0, "split"] = "holdout"
    with pytest.raises(ValueError):
        verify_no_leakage(labels, splits)


def test_summarize_totals_and_order():
    labels = _synthetic_labels()
    splits = make_star_splits(labels)
    s = summarize(labels, splits)
    assert list(s["split"]) == list(SPLIT_NAMES)
    assert s["stars"].sum() == labels["kepid"].nunique()
    assert s["tces"].sum() == len(labels)
    assert s["planet"].sum() == labels["label"].sum()
