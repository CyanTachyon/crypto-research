import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from crypto_transformer.data.collectors.cache import load_cache, save_cache

PREFIX = "binance"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FETCH_LIMIT = 1000
RATE_LIMIT_PAUSE = 0.5


def _ohlcv_to_df(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start_date: str = "2020-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    if use_cache:
        cached = load_cache(PREFIX, symbol, timeframe)
        if cached is not None and len(cached) > 0:
            last_ts = pd.Timestamp(cached["timestamp"].iloc[-1])
            if (datetime.now(timezone.utc) - last_ts).days < 1:
                print(f"[cache hit] {symbol} {timeframe}: {len(cached)} rows")
                return cached
            print(f"[cache partial] {symbol} {timeframe}: {len(cached)} rows, fetching from {last_ts}")
            return _fetch_incremental(symbol, timeframe, cached)

    return _fetch_full(symbol, timeframe, start_date)


def _fetch_full(symbol: str, timeframe: str, start_date: str) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    all_rows = []

    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=FETCH_LIMIT)
        if not rows:
            break
        all_rows.extend(rows)
        since = rows[-1][0] + 1
        print(f"  fetched {len(all_rows)} rows for {symbol}")
        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)

    df = _ohlcv_to_df(all_rows)
    path = save_cache(df, PREFIX, symbol, timeframe)
    print(f"[saved] {symbol} {timeframe}: {len(df)} rows -> {path}")
    return df


def _fetch_incremental(symbol: str, timeframe: str, cached: pd.DataFrame) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    last_ts_ms = int(pd.Timestamp(cached["timestamp"].iloc[-1]).timestamp() * 1000) + 1
    since = last_ts_ms
    all_rows = []

    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=FETCH_LIMIT)
        if not rows:
            break
        all_rows.extend(rows)
        since = rows[-1][0] + 1
        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return cached

    new_df = _ohlcv_to_df(all_rows)
    combined = pd.concat([cached, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    path = save_cache(combined, PREFIX, symbol, timeframe)
    print(f"[updated] {symbol} {timeframe}: {len(combined)} rows (added {len(new_df)}) -> {path}")
    return combined
