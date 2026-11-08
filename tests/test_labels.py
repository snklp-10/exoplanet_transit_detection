import pandas as pd
import pytest

from exotransit.datasets.labels import make_labels


def _toy():
    return pd.DataFrame(
        {
            "kepid": [3, 1, 2, 2, 4],
            "tce_plnt_num": [1, 1, 1, 2, 1],
            "av_training_set": ["PC", "AFP", "NTP", "UNK", "PC"],
            "tce_period": [10.0, 5.0, 3.0, 8.0, 20.0],
            "tce_time0bk": [100.0, 110.0, 120.0, 130.0, 140.0],
            "tce_duration": [24.0, 12.0, 6.0, 12.0, 48.0],  # hours
            "tce_depth": [500.0, 800.0, 100.0, 300.0, 900.0],
            "tce_model_snr": [10.0, 20.0, 8.0, 9.0, 30.0],
            "tce_quarters": ["1111", "1111", "1111", "1111", "1111"],
            "tce_num_transits": [10, 20, 15, 12, 5],
        }
    )


def test_labels_and_unk_split():
    labeled, unk = make_labels(_toy())
    # labeled has 4 rows, unk has 1; PC rows have label 1, AFP/NTP have 0
    assert len(labeled) == 4
    assert len(unk) == 1
    assert labeled.set_index("kepid")["label"].to_dict() == {1: 0, 2: 0, 3: 1, 4: 1}
    assert (unk["disposition"] == "UNK").all()
    assert "label" not in unk.columns


def test_duration_converted_to_days():
    labeled, _ = make_labels(_toy())
    # the PC row with 24 hours has duration_days == 1.0
    row = labeled[labeled["kepid"] == 3].iloc[0]
    assert row["duration_day"] == pytest.approx(1.0)


def test_sorted_by_kepid_then_plnt_num():
    toy = _toy()
    # kepid column is non-decreasing
    toy.loc[len(toy)] = [3, 2, "AFP", 7.0, 150.0, 12.0, 200.0, 12.0, "1111", 9]
    toy = toy.iloc[::-1]  # scramble the row order
    labeled, _ = make_labels(toy)
    keys = list(zip(labeled["kepid"], labeled["tce_plnt_num"]))  # noqa: B905
    assert keys == sorted(keys)


def test_unexpected_disposition_raises():
    bad = _toy()
    bad.loc[0, "av_training_set"] = "XYZ"
    # make_labels should raise AssertionError (hint: pytest.raises)
    with pytest.raises(AssertionError):
        make_labels(bad)


def test_placeholders_become_nan_and_row_is_kept():
    toy = _toy()
    toy.loc[0, ["tce_model_snr", "tce_depth", "tce_num_transits"]] = [-1.0, 0.0, 0]
    labeled, unk = make_labels(toy)
    assert len(labeled) + len(unk) == len(toy)
    row = labeled[labeled["kepid"] == 3].iloc[0]
    assert row["fit_failed"]
    assert row[["snr", "depth_ppm", "tce_num_transits"]].isna().all()


def test_normal_rows_are_not_flagged():
    labeled, unk = make_labels(_toy())
    assert not labeled["fit_failed"].any()
    assert not unk["fit_failed"].any()


def test_missing_period_raises():
    bad = _toy()
    bad.loc[0, "tce_period"] = float("nan")
    with pytest.raises(ValueError):
        make_labels(bad)
