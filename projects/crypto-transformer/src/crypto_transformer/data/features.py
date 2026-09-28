import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = _default_config()

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    rsi_period = config.get("rsi_period", 14)
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)

    macd_fast = config.get("macd_fast", 12)
    macd_slow = config.get("macd_slow", 26)
    macd_signal = config.get("macd_signal", 9)
    macd_result = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_result is not None and not macd_result.empty:
        col_prefix = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        df["macd"] = macd_result.get(f"{col_prefix}", 0)
        df["macd_signal"] = macd_result.get(f"{col_prefix}_s", 0)
        df["macd_hist"] = macd_result.get(f"{col_prefix}_h", 0)
    else:
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0

    atr_period = config.get("atr_period", 14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)

    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.0)
    bb = ta.bbands(df["close"], length=bb_period, std=bb_std)
    if bb is not None and not bb.empty:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    else:
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = df["bb_pct_b"] = np.nan

    df["obv"] = ta.obv(df["close"], df["volume"])

    vol_ma_period = config.get("volume_ma_period", 20)
    df["volume_ma"] = df["volume"].rolling(vol_ma_period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]

    if config.get("atr_normalize", True):
        atr_safe = df["atr"].replace(0, np.nan)
        for col in ["open", "high", "low", "close", "bb_upper", "bb_mid", "bb_lower"]:
            if col in df.columns:
                df[col] = df[col] / atr_safe

    warmup = max(rsi_period, macd_slow + macd_signal, atr_period, bb_period, vol_ma_period) + 5
    df = df.iloc[warmup:].reset_index(drop=True)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_return", "rsi",
        "macd", "macd_signal", "macd_hist",
        "atr", "bb_upper", "bb_mid", "bb_lower", "bb_pct_b",
        "obv", "volume_ratio",
    ]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "log_return", "rsi",
    "macd", "macd_signal", "macd_hist",
    "atr", "bb_upper", "bb_mid", "bb_lower", "bb_pct_b",
    "obv", "volume_ratio",
]


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
