import time

import pandas as pd
import requests

from crypto_transformer.data.collectors.cache import load_cache, save_cache

PREFIX = "geckoterminal"
BASE_URL = "https://api.geckoterminal.com/api/v2"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
RATE_LIMIT_PAUSE = 3.5


def _parse_ohlcv(data: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def fetch_ohlcv(
    pool_address: str,
    timeframe: str = "hour",
    aggregate: int = 1,
    limit: int = 1000,
    currency: str = "usd",
    max_pages: int = 100,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_key = f"{pool_address}_{timeframe}_{aggregate}"
    if use_cache:
        cached = load_cache(PREFIX, cache_key, timeframe)
        if cached is not None and len(cached) > 0:
            print(f"[cache hit] {cache_key}: {len(cached)} rows")
            return cached

    all_rows = []
    before_timestamp = None

    for page in range(max_pages):
        params = {
            "aggregate": aggregate,
            "limit": limit,
            "currency": currency,
        }
        if before_timestamp is not None:
            params["before_timestamp"] = before_timestamp

        url = f"{BASE_URL}/networks/base/pools/{pool_address}/ohlcv/{timeframe}"
        for attempt in range(5):
            resp = requests.get(url, params=params, headers={"Accept": "application/json"})
            if resp.status_code == 429:
                wait = RATE_LIMIT_PAUSE * (attempt + 2)
                print(f"  rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()

        ohlcv_list = resp.json()["data"]["attributes"]["ohlcv_list"]
        if not ohlcv_list:
            break

        all_rows.extend(ohlcv_list)
        before_timestamp = ohlcv_list[0][0] - 1
        print(f"  page {page + 1}: fetched {len(ohlcv_list)} candles (total: {len(all_rows)})")
        time.sleep(RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)

    df = _parse_ohlcv(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    path = save_cache(df, PREFIX, cache_key, timeframe)
    print(f"[saved] {cache_key}: {len(df)} rows -> {path}")
    return df
