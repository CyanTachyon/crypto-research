from pathlib import Path

import pandas as pd

from crypto_transformer import RAW_DIR


def _cache_path(prefix: str, symbol: str, timeframe: str) -> Path:
    safe_name = symbol.replace("/", "_")
    return RAW_DIR / f"{prefix}_{safe_name}_{timeframe}.parquet"


def load_cache(prefix: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(prefix, symbol, timeframe)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_cache(df: pd.DataFrame, prefix: str, symbol: str, timeframe: str) -> Path:
    path = _cache_path(prefix, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
