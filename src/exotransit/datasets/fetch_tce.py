from pathlib import Path

import pandas as pd
import pyvo

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"

COLS = [
    # identity and split key
    "kepid",
    "tce_plnt_num",
    # ephemeris for detrending and folding
    "tce_period",  # days
    "tce_time0bk",  # days, BKJD
    "tce_duration",  # HOURS
    # label source
    "av_training_set",
    # reference values
    "tce_depth",  # ppm
    "tce_model_snr",
    # useful for the download step and filtering
    "tce_quarters",
    "tce_num_transits",
]


def fetch_tce_table(cache_path: Path, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    # return (dataframe, source) where source is 'cache' or 'downloaded'.

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        return df, "cache"

    query = f"select {', '.join(COLS)} from q1_q17_dr24_tce"
    tap = pyvo.dal.TAPService(TAP_URL)
    df = tap.search(query).to_table().to_pandas()  # type: ignore

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df, "downloaded"


# print(len(df))                              # expect 20367
# print(df["av_training_set"].value_counts()) # expect AFP 9596, UNK 4630, PC 3600, NTP 2541
# print(df.isna().sum()) #missing values per column
# print(df)
