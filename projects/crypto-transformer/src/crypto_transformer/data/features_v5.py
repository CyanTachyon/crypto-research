import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path


FEATURE_COLUMNS_V5 = [
    "log_return",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr_norm", "bb_pct_b", "volume_ratio",
    "return_5d", "return_10d", "return_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "high_low_range_5d", "high_low_range_20d",
    "dow_sin", "dow_cos",
    "btc_corr_20d",
    "close_norm", "volume_norm",
    "return_rank_10d",
    "vol_ratio_5_20",
    "obv_slope_10d",
    "momentum_roc_10d",
    "funding_rate_avg",
    "funding_rate_ma7",
    "funding_rate_zscore",
    "funding_rate_extreme",
    "fear_greed",
    "fear_greed_ma7",
    "fear_greed_zscore",
    "fear_greed_extreme_fear",
    "fear_greed_extreme_greed",
    "btc_dominance",
    "btc_dominance_change_5d",
    "btc_dominance_change_20d",
    "hashrate_change_7d",
    "hashrate_change_30d",
    "active_addresses_change_7d",
    "active_addresses_ma7_ratio",
    "tvl_change_7d",
    "tvl_ma30_ratio",
]

NUM_FEATURES_V5 = len(FEATURE_COLUMNS_V5)
EXTERNAL_DIR = Path("data/external")


def _load_external():
    data = {}
    files = {
        "fear_greed": "fear_greed.parquet",
        "btc_dominance": "btc_dominance.parquet",
        "onchain": "blockchain_onchain.parquet",
        "defi_tvl": "defi_tvl.parquet",
    }
    for key, fname in files.items():
        p = EXTERNAL_DIR / fname
        if p.exists():
            data[key] = pd.read_parquet(p)
        else:
            data[key] = pd.DataFrame()

    funding = {}
    for pair_sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                     "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "UNIUSDT",
                     "OPUSDT", "AAVEUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT"]:
        p = EXTERNAL_DIR / f"funding_rate_{pair_sym}.parquet"
        if p.exists():
            funding[pair_sym] = pd.read_parquet(p)
    data["funding"] = funding
    return data


def _map_pair_to_funding_symbol(pair_name: str) -> str:
    return pair_name.replace("/", "").upper()


def compute_features_v5(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    cross_sectional_ranks: pd.Series | None = None,
    config: dict | None = None,
    external_data: dict | None = None,
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
    if cross_sectional_ranks is not None and len(cross_sectional_ranks) == len(df):
        df["return_rank_10d"] = cross_sectional_ranks.values
    else:
        df["return_rank_10d"] = 0.5
    df["vol_ratio_5_20"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)
    obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["obv_slope_10d"] = obv.diff(10) / (obv.rolling(10).std().replace(0, np.nan))
    df["momentum_roc_10d"] = df["close"] / df["close"].shift(10) - 1

    df["funding_rate_avg"] = 0.0
    df["funding_rate_ma7"] = 0.0
    df["funding_rate_zscore"] = 0.0
    df["funding_rate_extreme"] = 0.0
    df["fear_greed"] = 50.0
    df["fear_greed_ma7"] = 50.0
    df["fear_greed_zscore"] = 0.0
    df["fear_greed_extreme_fear"] = 0.0
    df["fear_greed_extreme_greed"] = 0.0
    df["btc_dominance"] = 50.0
    df["btc_dominance_change_5d"] = 0.0
    df["btc_dominance_change_20d"] = 0.0
    df["hashrate_change_7d"] = 0.0
    df["hashrate_change_30d"] = 0.0
    df["active_addresses_change_7d"] = 0.0
    df["active_addresses_ma7_ratio"] = 1.0
    df["tvl_change_7d"] = 0.0
    df["tvl_ma30_ratio"] = 1.0

    if external_data is not None:
        dates_str = df["timestamp"].dt.strftime("%Y-%m-%d")

        def _map_ext(ext_df, col):
            if ext_df is None or ext_df.empty or col not in ext_df.columns:
                return
            ed = ext_df.copy()
            ed["_ds"] = pd.to_datetime(ed["date"]).dt.strftime("%Y-%m-%d")
            m = dict(zip(ed["_ds"], ed[col]))
            df[col] = dates_str.map(m).values

        funding_sym = _map_pair_to_funding_symbol(df.get("pair", pd.Series(["BTCUSDT"])).iloc[0] if "pair" in df.columns else "BTC/USDT")
        funding_data = external_data.get("funding", {}).get(funding_sym)
        if funding_data is not None and not funding_data.empty:
            fr = funding_data.copy()
            fr["date"] = pd.to_datetime(fr["date"]).dt.normalize()
            fr = fr.set_index("date")
            fr["funding_rate_ma7"] = fr["funding_rate_avg"].rolling(7, min_periods=1).mean()
            fr_mean = fr["funding_rate_avg"].rolling(30, min_periods=1).mean()
            fr_std = fr["funding_rate_avg"].rolling(30, min_periods=1).std().replace(0, np.nan)
            fr["funding_rate_zscore"] = ((fr["funding_rate_avg"] - fr_mean) / fr_std).fillna(0)
            fr["funding_rate_extreme"] = (fr["funding_rate_avg"].abs() > 0.001).astype(float)
            fr = fr.reset_index()
            for col in ["funding_rate_avg", "funding_rate_ma7", "funding_rate_zscore", "funding_rate_extreme"]:
                if col in fr.columns:
                    _map_ext(fr, col)

        fg = external_data.get("fear_greed")
        if fg is not None and not fg.empty:
            fg_c = fg.copy()
            fg_c["date"] = pd.to_datetime(fg_c["date"]).dt.normalize()
            fg_c = fg_c.set_index("date")
            fg_c["fear_greed_ma7"] = fg_c["fear_greed"].rolling(7, min_periods=1).mean()
            fg_mean = fg_c["fear_greed"].rolling(30, min_periods=1).mean()
            fg_std = fg_c["fear_greed"].rolling(30, min_periods=1).std().replace(0, np.nan)
            fg_c["fear_greed_zscore"] = ((fg_c["fear_greed"] - fg_mean) / fg_std).fillna(0)
            fg_c["fear_greed_extreme_fear"] = (fg_c["fear_greed"] < 20).astype(float)
            fg_c["fear_greed_extreme_greed"] = (fg_c["fear_greed"] > 80).astype(float)
            fg_c = fg_c.reset_index()
            for col in ["fear_greed", "fear_greed_ma7", "fear_greed_zscore", "fear_greed_extreme_fear", "fear_greed_extreme_greed"]:
                if col in fg_c.columns:
                    _map_ext(fg_c, col)

        dom = external_data.get("btc_dominance")
        if dom is not None and not dom.empty:
            dom_c = dom.copy()
            dom_c["date"] = pd.to_datetime(dom_c["date"]).dt.normalize()
            dom_c = dom_c.set_index("date")
            if "btc_dominance" in dom_c.columns:
                dom_c["btc_dominance_change_5d"] = dom_c["btc_dominance"].pct_change(5)
                dom_c["btc_dominance_change_20d"] = dom_c["btc_dominance"].pct_change(20)
                dom_c = dom_c.reset_index()
                for col in ["btc_dominance", "btc_dominance_change_5d", "btc_dominance_change_20d"]:
                    if col in dom_c.columns:
                        _map_ext(dom_c, col)

        onchain = external_data.get("onchain")
        if onchain is not None and not onchain.empty:
            oc = onchain.copy()
            oc["date"] = pd.to_datetime(oc["date"]).dt.normalize()
            oc = oc.set_index("date")
            if "hashrate" in oc.columns:
                oc["hashrate_change_7d"] = oc["hashrate"].pct_change(7)
                oc["hashrate_change_30d"] = oc["hashrate"].pct_change(30)
                oc = oc.reset_index()
                for col in ["hashrate_change_7d", "hashrate_change_30d"]:
                    if col in oc.columns:
                        _map_ext(oc, col)
            if "active_addresses" in onchain.columns:
                oc2 = onchain.copy()
                oc2["date"] = pd.to_datetime(oc2["date"]).dt.normalize()
                oc2 = oc2.set_index("date")
                oc2["active_addresses_change_7d"] = oc2["active_addresses"].pct_change(7)
                oc2["active_addresses_ma7"] = oc2["active_addresses"].rolling(7, min_periods=1).mean()
                oc2["active_addresses_ma7_ratio"] = oc2["active_addresses"] / oc2["active_addresses_ma7"].replace(0, np.nan)
                oc2 = oc2.reset_index()
                for col in ["active_addresses_change_7d", "active_addresses_ma7_ratio"]:
                    if col in oc2.columns:
                        _map_ext(oc2, col)

        tvl = external_data.get("defi_tvl")
        if tvl is not None and not tvl.empty:
            tvl_c = tvl.copy()
            tvl_c["date"] = pd.to_datetime(tvl_c["date"]).dt.normalize()
            tvl_c = tvl_c.set_index("date")
            if "total_tvl" in tvl_c.columns:
                tvl_c["tvl_change_7d"] = tvl_c["total_tvl"].pct_change(7)
                tvl_c["tvl_ma30"] = tvl_c["total_tvl"].rolling(30, min_periods=1).mean()
                tvl_c["tvl_ma30_ratio"] = tvl_c["total_tvl"] / tvl_c["tvl_ma30"].replace(0, np.nan)
                tvl_c = tvl_c.reset_index()
                for col in ["tvl_change_7d", "tvl_ma30_ratio"]:
                    if col in tvl_c.columns:
                        _map_ext(tvl_c, col)

    df["close_raw"] = df["close"].copy()
    warmup = 30
    df = df.iloc[warmup:].reset_index(drop=True)
    df[FEATURE_COLUMNS_V5] = df[FEATURE_COLUMNS_V5].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS_V5).reset_index(drop=True)
    return df


def compute_cross_sectional_features(
    all_dfs: dict[str, pd.DataFrame],
    lookback: int = 10,
) -> dict[str, pd.Series]:
    pair_returns = {}
    for name, df in all_dfs.items():
        ts_index = df["timestamp"].values
        ret = df["close"].pct_change(lookback)
        pair_returns[name] = pd.Series(ret.values, index=ts_index)

    all_timestamps = sorted(set().union(*(set(s.index) for s in pair_returns.values())))
    rank_results = {}
    for name in all_dfs:
        ranks = []
        ret_s = pair_returns[name]
        for ts in all_timestamps:
            vals = []
            for other_name, other_s in pair_returns.items():
                if ts in other_s.index:
                    vals.append(other_s.loc[ts])
            if ts in ret_s.index and len(vals) >= 5:
                vals_arr = np.array(vals, dtype=float)
                vals_arr = vals_arr[~np.isnan(vals_arr)]
                rank = np.mean(ret_s.loc[ts] > vals_arr) if len(vals_arr) >= 3 else 0.5
            else:
                rank = 0.5
            ranks.append(rank)
        rank_series = pd.Series(ranks, index=all_timestamps)
        df_ts = all_dfs[name]["timestamp"].values
        aligned = rank_series.reindex(df_ts, method="nearest", tolerance=np.timedelta64(1, "D"))
        rank_results[name] = aligned.fillna(0.5).reset_index(drop=True)
    return rank_results
