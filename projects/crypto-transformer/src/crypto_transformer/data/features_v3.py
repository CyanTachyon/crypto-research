import numpy as np
import pandas as pd
import pandas_ta as ta


FEATURE_COLUMNS_V3 = [
    "log_return",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr_norm", "bb_pct_b", "volume_ratio",
    "return_5d", "return_10d", "return_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "high_low_range_5d", "high_low_range_20d",
    "dow_sin", "dow_cos",
    "btc_corr_20d",
    "close_norm", "volume_norm",
]

NUM_FEATURES_V3 = len(FEATURE_COLUMNS_V3)


def compute_features_v3(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    if config is None:
        config = {}

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    rsi_p = config.get("rsi_period", 14)
    df["rsi"] = ta.rsi(df["close"], length=rsi_p)

    mf, ms, msig = config.get("macd_fast", 12), config.get("macd_slow", 26), config.get("macd_signal", 9)
    macd_r = ta.macd(df["close"], fast=mf, slow=ms, signal=msig)
    pfx = f"MACD_{mf}_{ms}_{msig}"
    df["macd"] = macd_r.get(f"{pfx}", 0.0) if macd_r is not None else 0.0
    df["macd_signal"] = macd_r.get(f"{pfx}_s", 0.0) if macd_r is not None else 0.0
    df["macd_hist"] = macd_r.get(f"{pfx}_h", 0.0) if macd_r is not None else 0.0

    atr_p = config.get("atr_period", 14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_p)
    atr_safe = df["atr"].replace(0, np.nan)
    df["atr_norm"] = df["atr"] / df["close"]

    bb_p, bb_s = config.get("bb_period", 20), config.get("bb_std", 2.0)
    bb = ta.bbands(df["close"], length=bb_p, std=bb_s)
    if bb is not None and not bb.empty:
        df["bb_pct_b"] = (df["close"] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    else:
        df["bb_pct_b"] = np.nan

    vma = config.get("volume_ma_period", 20)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(vma).mean()

    for lag, name in [(5, "5d"), (10, "10d"), (20, "20d")]:
        df[f"return_{name}"] = np.log(df["close"] / df["close"].shift(lag))

    for w, name in [(5, "5d"), (10, "10d"), (20, "20d")]:
        df[f"vol_{name}"] = df["log_return"].rolling(w).std()

    df["high_low_range_5d"] = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / df["close"]
    df["high_low_range_20d"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["close"]

    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    if btc_df is not None and len(btc_df) >= len(df):
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
        own_ret = df["log_return"]
        min_len = min(len(btc_ret), len(own_ret))
        df["btc_corr_20d"] = np.nan
        btc_vals = btc_ret.values[:min_len]
        own_vals = own_ret.values[:min_len]
        for i in range(20, min_len):
            df.loc[i, "btc_corr_20d"] = np.corrcoef(btc_vals[i-20:i], own_vals[i-20:i])[0, 1]
        df["btc_corr_20d"] = df["btc_corr_20d"].fillna(0)
    else:
        df["btc_corr_20d"] = 0.0

    df["close_norm"] = df["close"] / atr_safe
    df["volume_norm"] = df["volume"] / df["volume"].rolling(20).mean()
    df["close_raw"] = df["close"].copy()

    warmup = 30
    df = df.iloc[warmup:].reset_index(drop=True)

    df[FEATURE_COLUMNS_V3] = df[FEATURE_COLUMNS_V3].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS_V3).reset_index(drop=True)
    return df
