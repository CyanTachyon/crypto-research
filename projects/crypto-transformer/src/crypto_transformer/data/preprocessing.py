import numpy as np
import pandas as pd

from crypto_transformer.data.features import compute_features, FEATURE_COLUMNS


def add_raw_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close_raw"] = df["close"].copy()
    return df


def prepare_dataframe(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    df = add_raw_close(df)
    df = compute_features(df, config)
    return df
