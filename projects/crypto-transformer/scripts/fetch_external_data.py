import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data/external")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RETRY = 3


def _get(url, params=None, timeout=15):
    for attempt in range(RETRY):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == RETRY - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_funding_rate(symbol="BTCUSDT", start="2020-01-01"):
    path = DATA_DIR / f"funding_rate_{symbol}.parquet"
    if path.exists():
        print(f"  [cache] funding_rate_{symbol}")
        return pd.read_parquet(path)

    print(f"  Fetching funding rate {symbol}...")
    all_data = []
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now().timestamp() * 1000)
    cursor = start_ms

    while cursor < end_ms:
        data = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                    {"symbol": symbol, "startTime": cursor, "limit": 1000})
        if not data:
            break
        for d in data:
            all_data.append({
                "timestamp": pd.Timestamp(int(d["fundingTime"]), unit="ms"),
                "funding_rate": float(d["fundingRate"]),
            })
        cursor = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data)
    if df.empty:
        return df

    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date").agg(
        funding_rate_avg=("funding_rate", "mean"),
        funding_rate_max=("funding_rate", "max"),
        funding_rate_min=("funding_rate", "min"),
        funding_rate_count=("funding_rate", "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_parquet(path, index=False)
    print(f"    Saved {len(daily)} days to {path}")
    return daily


def fetch_fear_greed():
    path = DATA_DIR / "fear_greed.parquet"
    if path.exists():
        print(f"  [cache] fear_greed")
        return pd.read_parquet(path)

    print("  Fetching Fear & Greed Index...")
    data = _get("https://api.alternative.me/fng/", {"limit": 0, "format": "json"})
    records = []
    for d in data["data"]:
        records.append({
            "date": pd.Timestamp(int(d["timestamp"]), unit="s"),
            "fear_greed": int(d["value"]),
            "fear_greed_class": d["value_classification"],
        })
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"    Saved {len(df)} days to {path}")
    return df


def fetch_btc_dominance(days=2000):
    path = DATA_DIR / "btc_dominance.parquet"
    if path.exists():
        print(f"  [cache] btc_dominance")
        return pd.read_parquet(path)

    print("  Computing BTC dominance from Binance OHLCV + CoinGecko...")
    from crypto_transformer.data.collectors.binance import fetch_ohlcv
    btc_ohlcv = fetch_ohlcv("BTC/USDT", "1d")

    try:
        global_data = _get("https://api.coingecko.com/api/v3/global", timeout=10)
        current_dom = global_data["data"]["market_cap_percentage"]["btc"]
    except Exception:
        current_dom = 56.0

    try:
        tvl_raw = _get("https://api.llama.fi/v2/historicalChainTvl", timeout=30)
        tvl_map = {}
        if tvl_raw and isinstance(tvl_raw, list):
            for d in tvl_raw:
                if isinstance(d, dict):
                    tvl_map[pd.Timestamp(d["date"], unit="s").normalize()] = d["tvl"]
    except Exception:
        tvl_map = {}

    records = []
    for _, row in btc_ohlcv.iterrows():
        rec = {
            "date": pd.Timestamp(row["timestamp"]).normalize(),
            "btc_price": row["close"],
        }
        mcap_est = row["close"] * 19.8e6
        rec["btc_market_cap"] = mcap_est
        rec["btc_dominance"] = current_dom
        if rec["date"] in tvl_map:
            rec["total_tvl_estimate"] = tvl_map[rec["date"]]
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"    Saved {len(df)} days (dominance={current_dom:.0f}%) to {path}")
    return df


def fetch_blockchain_charts():
    path = DATA_DIR / "blockchain_onchain.parquet"
    if path.exists():
        print(f"  [cache] blockchain_onchain")
        return pd.read_parquet(path)

    print("  Fetching blockchain on-chain data...")
    print("    Note: Blockchain.com charts API changed, using limited stats endpoint")
    print("    Skipping historical on-chain for now (hashrate/addresses)")

    try:
        stats = _get("https://api.blockchain.info/stats", timeout=10)
        print(f"    Current stats: hashrate={stats.get('hash_rate', 0):.0f}")
    except Exception:
        pass

    df = pd.DataFrame(columns=["date"])
    df.to_parquet(path, index=False)
    print(f"    Saved empty on-chain file (will use 0 for missing features)")
    return df


def fetch_defi_tvl():
    path = DATA_DIR / "defi_tvl.parquet"
    if path.exists():
        print(f"  [cache] defi_tvl")
        return pd.read_parquet(path)

    print("  Fetching DeFi TVL from DefiLlama...")
    data = _get("https://api.llama.fi/v2/historicalChainTvl", timeout=30)
    if not data:
        return pd.DataFrame()

    records = []
    for d in data:
        if isinstance(d, dict) and "date" in d and "tvl" in d:
            records.append({
                "date": pd.Timestamp(d["date"], unit="s").normalize(),
                "total_tvl": d["tvl"],
            })

    df = pd.DataFrame(records)
    df = df.groupby("date").sum().reset_index()
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"    Saved {len(df)} days to {path}")
    return df


def fetch_funding_multi(symbols=None):
    if symbols is None:
        symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "UNIUSDT",
            "OPUSDT", "AAVEUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
        ]
    results = {}
    for sym in symbols:
        results[sym] = fetch_funding_rate(sym)
        time.sleep(0.3)
    return results


def main():
    print("=" * 60)
    print("Fetching external data sources for V5")
    print("=" * 60)

    print("\n[1/5] Funding rates (15 pairs)...")
    funding = fetch_funding_multi()

    print("\n[2/5] Fear & Greed Index...")
    fg = fetch_fear_greed()

    print("\n[3/5] BTC Dominance & Market Cap...")
    dom = fetch_btc_dominance()

    print("\n[4/5] Blockchain on-chain (hashrate, addresses, etc)...")
    onchain = fetch_blockchain_charts()

    print("\n[5/5] DeFi TVL...")
    tvl = fetch_defi_tvl()

    print("\n" + "=" * 60)
    print("Summary:")
    for name, df in [
        ("Funding Rate (BTC)", funding.get("BTCUSDT", pd.DataFrame())),
        ("Fear & Greed", fg),
        ("BTC Dominance", dom),
        ("On-chain", onchain),
        ("DeFi TVL", tvl),
    ]:
        if df is not None and not df.empty:
            print(f"  {name:20s}: {len(df):5d} rows  {df['date'].min().date()} ~ {df['date'].max().date()}")
        else:
            print(f"  {name:20s}: EMPTY")

    print("\nAll data cached in data/external/")


if __name__ == "__main__":
    main()
