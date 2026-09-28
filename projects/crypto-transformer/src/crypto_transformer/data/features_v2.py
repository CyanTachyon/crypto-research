import numpy as np
import pandas as pd
import pandas_ta as ta


FEATURE_COLUMNS_V2 = [
    "log_return", "rsi", "macd", "macd_signal", "macd_hist",
    "atr", "bb_pct_b", "volume_ratio",
    "return_4h", "return_12h", "return_24h", "return_48h",
    "vol_4h", "vol_12h", "vol_24h",
    "high_low_range_4h", "high_low_range_24h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "cross_return", "cross_vol_ratio",
    "close_norm", "volume_norm",
]

NUM_FEATURES_V2 = len(FEATURE_COLUMNS_V2)


def _default_config() -> dict:
    return {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "volume_ma_period": 20,
        "atr_normalize": True,
    }


def compute_features_v2(
    df: pd.DataFrame,
    cross_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    if config is None:
        config = _default_config()

    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["rsi"] = ta.rsi(df["close"], length=config["rsi_period"])

    macd_result = ta.macd(
        df["close"],
        fast=config["macd_fast"],
        slow=config["macd_slow"],
        signal=config["macd_signal"],
    )
    prefix = f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    df["macd"] = macd_result.get(f"{prefix}", 0.0) if macd_result is not None else 0.0
    df["macd_signal"] = macd_result.get(f"{prefix}_s", 0.0) if macd_result is not None else 0.0
    df["macd_hist"] = macd_result.get(f"{prefix}_h", 0.0) if macd_result is not None else 0.0

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=config["atr_period"])

    bb = ta.bbands(df["close"], length=config["bb_period"], std=config["bb_std"])
    if bb is not None and not bb.empty:
        df["bb_pct_b"] = (df["close"] - bb.iloc[:, 2]) / (bb.iloc[:, 0] - bb.iloc[:, 2])
    else:
        df["bb_pct_b"] = np.nan

    vol_ma = df["volume"].rolling(config["volume_ma_period"]).mean()
    df["volume_ratio"] = df["volume"] / vol_ma

    for lag, name in [(4, "4h"), (12, "12h"), (24, "24h"), (48, "48h")]:
        df[f"return_{name}"] = np.log(df["close"] / df["close"].shift(lag))

    for window, name in [(4, "4h"), (12, "12h"), (24, "24h")]:
        df[f"vol_{name}"] = df["log_return"].rolling(window).std()

    df["high_low_range_4h"] = (df["high"].rolling(4).max() - df["low"].rolling(4).min()) / df["close"]
    df["high_low_range_24h"] = (df["high"].rolling(24).max() - df["low"].rolling(24).min()) / df["close"]

    hours = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    if cross_df is not None:
        cross_returns = np.log(cross_df["close"] / cross_df["close"].shift(1))
        df["cross_return"] = cross_returns.values[: len(df)] if len(cross_returns) >= len(df) else np.nan
        cross_vol = cross_returns.rolling(24).std()
        own_vol = df["log_return"].rolling(24).std()
        df["cross_vol_ratio"] = (cross_vol / own_vol).values[: len(df)] if len(cross_vol) >= len(df) else np.nan
    else:
        df["cross_return"] = 0.0
        df["cross_vol_ratio"] = 1.0

    if config.get("atr_normalize", True):
        atr_safe = df["atr"].replace(0, np.nan)
        df["close_norm"] = df["close"] / atr_safe
    else:
        df["close_norm"] = df["close"] / df["close"].iloc[0]

    df["volume_norm"] = df["volume"] / df["volume"].rolling(48).mean()

    df["close_raw"] = df["close"].copy()

    warmup = 56
    df = df.iloc[warmup:].reset_index(drop=True)

    df[FEATURE_COLUMNS_V2] = df[FEATURE_COLUMNS_V2].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS_V2).reset_index(drop=True)

    return df
