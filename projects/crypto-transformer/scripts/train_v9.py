import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_v4 as v4  # noqa: E402
from crypto_transformer.data.features_v4 import (  # noqa: E402
    FEATURE_COLUMNS_V4,
    NUM_FEATURES_V4,
    compute_cross_sectional_features,
    compute_features_v4,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PAIRS = v4.PAIRS
SEQ_LEN = v4.SEQ_LEN
HORIZON = v4.HORIZON
TRAIN_CUTOFF = v4.TRAIN_CUTOFF
SEEDS = v4.SEEDS
NUM_PAIRS = len(PAIRS)
INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001
MIN_TRAIN_DAYS = 90
TRADE_WINDOW_DAYS = 30
POS_SIZE = 1.0 / NUM_PAIRS
POLICY_EPOCHS = 120
META_EPOCHS = 90
BANDIT_EPOCHS = 90
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "docs" / "figures"
DOC_DIR = ROOT / "docs"
PRED_PARQUET = DATA_DIR / "v9_v4_predictions.parquet"
PRED_CSV = DATA_DIR / "v9_v4_predictions.csv"
ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    print("=" * 80)
    print("V9: V4 signal trading-layer optimization + meta/bandit/policy search")
    print("=" * 80)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V9 neural training; CPU fallback is forbidden.")
    count = torch.cuda.device_count()
    for i in range(count):
        print(f"  visible cuda:{i}: {torch.cuda.get_device_name(i)}")
    return torch.device("cuda")


def raw_cache_path(symbol: str, timeframe: str = "1d") -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return DATA_DIR / "raw" / f"binance_{safe}_{timeframe}.parquet"


def load_cached_ohlcv(symbol: str, timeframe: str = "1d") -> pd.DataFrame:
    path = raw_cache_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached OHLCV file: {path}")
    df = pd.read_parquet(path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def strip_tz(values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.to_datetime(values)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s.dt.normalize()


def build_test_index_from_cache() -> tuple[dict[str, pd.DataFrame], list[str], dict[str, int]]:
    print("\n[1/7] Building V4 test index from local parquet cache...")
    raw = {pair: load_cached_ohlcv(pair, "1d") for pair in PAIRS}
    btc = raw["BTC/USDT"]
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {
        pair: compute_features_v4(raw[pair], btc_df=btc, cross_sectional_ranks=ranks.get(pair))
        for pair in PAIRS
    }
    pair_names = sorted(all_dfs.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_concat = pd.concat(
        [all_dfs[p][all_dfs[p]["timestamp"].values < cutoff] for p in pair_names],
        ignore_index=True,
    )
    scaler = StandardScaler()
    scaler.fit(train_concat[FEATURE_COLUMNS_V4].values)

    test_dfs: dict[str, pd.DataFrame] = {}
    for pair in pair_names:
        df = all_dfs[pair][all_dfs[pair]["timestamp"].values >= cutoff].copy().reset_index(drop=True)
        df[FEATURE_COLUMNS_V4] = scaler.transform(df[FEATURE_COLUMNS_V4].values).astype(np.float32)
        test_dfs[pair] = df
    print(f"  pairs={len(pair_names)}, cutoff={TRAIN_CUTOFF}, seq_len={SEQ_LEN}, horizon={HORIZON}")
    return test_dfs, pair_names, pair_to_id


def load_v4_models(pair_names: list[str], device: torch.device) -> list[nn.Module]:
    print("\n[2/7] Loading V4 checkpoints for inference only...")
    models: list[nn.Module] = []
    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / f"v4_seed_{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing V4 checkpoint {ckpt}; V9 will not retrain V4.")
        model = v4.RegressionTransformer(nf=NUM_FEATURES_V4, np_=len(pair_names)).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"  loaded {ckpt.name}")
    return models


def run_v4_inference(
    models: list[nn.Module],
    test_dfs: dict[str, pd.DataFrame],
    pair_names: list[str],
    pair_to_id: dict[str, int],
    device: torch.device,
) -> pd.DataFrame:
    print("\n[3/7] Running V4 ensemble inference for V9...")
    records: list[dict[str, object]] = []
    batch_size = 256
    for pair_name in pair_names:
        df = test_dfs[pair_name]
        feat = df[FEATURE_COLUMNS_V4].values
        timestamps = df["timestamp"].values
        close = df["close_raw"].values.astype(float)
        pair_id = pair_to_id[pair_name]
        n = len(df) - SEQ_LEN - HORIZON
        if n <= 0:
            continue

        all_preds: list[float] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xs = np.array([feat[i : i + SEQ_LEN] for i in range(start, end)], dtype=np.float32)
            pids = np.full(end - start, pair_id, dtype=np.int64)
            x_t = torch.tensor(xs, dtype=torch.float32, device=device)
            pid_t = torch.tensor(pids, dtype=torch.long, device=device)
            preds = []
            with torch.no_grad():
                for model in models:
                    preds.append(model(x_t, pid_t).detach().cpu().numpy())
            all_preds.extend(np.mean(preds, axis=0).astype(float).tolist())

        for i, pred in enumerate(all_preds):
            di = i + SEQ_LEN
            next_ret = float(close[di + 1] / close[di] - 1) if di + 1 < len(close) else 0.0
            actual_3d = float(close[di + HORIZON] / close[di] - 1) if di + HORIZON < len(close) else 0.0
            records.append(
                {
                    "date": pd.Timestamp(timestamps[di]),
                    "pair": pair_name,
                    "pair_id": int(pair_id),
                    "pred_3d": float(pred),
                    "close": float(close[di]),
                    "next_1d_ret": next_ret,
                    "actual_3d_ret": actual_3d,
                }
            )
    pred_df = pd.DataFrame(records)
    pred_df["date"] = strip_tz(pred_df["date"])
    pred_df = pred_df.sort_values(["date", "pair"]).reset_index(drop=True)
    print(
        f"  predictions={len(pred_df)}, "
        f"period={pred_df['date'].min().date()} to {pred_df['date'].max().date()}"
    )
    return pred_df


def get_or_create_predictions(device: torch.device) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PRED_PARQUET.exists():
        print("\n[1/7] Loading cached V9 V4 predictions...")
        pred_df = pd.read_parquet(PRED_PARQUET)
        pred_df["date"] = strip_tz(pred_df["date"])
        print(
            f"  loaded predictions={len(pred_df)}, "
            f"period={pred_df['date'].min().date()} to {pred_df['date'].max().date()}"
        )
        return pred_df.sort_values(["date", "pair"]).reset_index(drop=True)

    test_dfs, pair_names, pair_to_id = build_test_index_from_cache()
    models = load_v4_models(pair_names, device)
    pred_df = run_v4_inference(models, test_dfs, pair_names, pair_to_id, device)
    pred_df.to_parquet(PRED_PARQUET, index=False)
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"  saved {PRED_PARQUET.relative_to(ROOT)} and {PRED_CSV.relative_to(ROOT)}")
    return pred_df


def add_state_features(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    print("\n[4/7] Building V9 state features without future leakage...")
    df = pred_df.copy().sort_values(["pair", "date"]).reset_index(drop=True)
    df["signal"] = np.sign(df["pred_3d"].values)
    df["abs_pred"] = df["pred_3d"].abs()
    df["pair_id_norm"] = df["pair_id"] / max(1, NUM_PAIRS - 1)

    df["pred_rank_pct"] = df.groupby("date")["pred_3d"].rank(pct=True)
    day_mean = df.groupby("date")["pred_3d"].transform("mean")
    day_std = df.groupby("date")["pred_3d"].transform("std").replace(0, np.nan)
    df["pred_z_cs"] = ((df["pred_3d"] - day_mean) / day_std).fillna(0.0)

    df["known_1d_ret"] = df.groupby("pair")["close"].pct_change().fillna(0.0)
    for window in [10, 20, 60]:
        df[f"vol_{window}"] = df.groupby("pair")["known_1d_ret"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 3)).std()
        )
        df[f"mom_{window}"] = df.groupby("pair")["close"].transform(
            lambda s, w=window: s / s.shift(w) - 1
        )

    roll_min = df.groupby("pair")["close"].transform(lambda s: s.rolling(20, min_periods=5).min())
    roll_max = df.groupby("pair")["close"].transform(lambda s: s.rolling(20, min_periods=5).max())
    df["price_pos_20"] = ((df["close"] - roll_min) / (roll_max - roll_min).replace(0, np.nan)).fillna(0.5)

    df["v4_signal_pnl"] = df["signal"] * df["next_1d_ret"]
    for window in [20, 60]:
        df[f"v4_edge_{window}"] = df.groupby("pair")["v4_signal_pnl"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(5, w // 4)).mean()
        )

    btc = df[df["pair"] == "BTC/USDT"].copy().sort_values("date")
    btc_maps: dict[str, pd.Series] = {}
    for col in ["mom_20", "mom_60", "vol_20", "known_1d_ret"]:
        btc_maps[col] = btc.set_index("date")[col]
    df["btc_mom_20"] = df["date"].map(btc_maps["mom_20"])
    df["btc_mom_60"] = df["date"].map(btc_maps["mom_60"])
    df["btc_vol_20"] = df["date"].map(btc_maps["vol_20"])
    df["btc_known_ret"] = df["date"].map(btc_maps["known_1d_ret"])

    market_known = df.groupby("date")["known_1d_ret"].mean().sort_index()
    market_mom_20 = market_known.rolling(20, min_periods=5).mean()
    market_vol_20 = market_known.rolling(20, min_periods=5).std()
    df["market_mom_20"] = df["date"].map(market_mom_20)
    df["market_vol_20"] = df["date"].map(market_vol_20)

    feature_cols = [
        "pred_3d",
        "abs_pred",
        "pred_rank_pct",
        "pred_z_cs",
        "signal",
        "pair_id_norm",
        "known_1d_ret",
        "vol_10",
        "vol_20",
        "vol_60",
        "mom_10",
        "mom_20",
        "mom_60",
        "price_pos_20",
        "v4_edge_20",
        "v4_edge_60",
        "btc_mom_20",
        "btc_mom_60",
        "btc_vol_20",
        "btc_known_ret",
        "market_mom_20",
        "market_vol_20",
    ]
    for col in feature_cols:
        med = df[col].replace([np.inf, -np.inf], np.nan).median()
        if not np.isfinite(med):
            med = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(float(med)).astype(float)
    print(f"  features={len(feature_cols)}, rows={len(df)}, dates={df['date'].nunique()}")
    return df.sort_values(["date", "pair"]).reset_index(drop=True), feature_cols


def fold_slices(dates: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    start = MIN_TRAIN_DAYS
    while start < len(dates):
        end = min(start + TRADE_WINDOW_DAYS, len(dates))
        folds.append((dates[:start], dates[start:end]))
        start = end
    return folds


def rank_score(metrics: dict[str, float | int | str]) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    ret = float(metrics.get("total_return_pct", 0.0)) / 100.0
    mdd = float(metrics.get("max_drawdown_pct", 0.0)) / 100.0
    trades = float(metrics.get("trades", 0.0))
    trade_penalty = min(trades / 100_000.0, 0.10)
    return sharpe + ret + mdd - trade_penalty


def simulate_positions(
    df: pd.DataFrame,
    positions: pd.Series | np.ndarray,
    name: str,
    dates: list[pd.Timestamp] | None = None,
    fee_rate: float = FEE_RATE,
) -> dict[str, object]:
    pos = pd.Series(np.asarray(positions, dtype=float), index=df.index)
    work = df[["date", "pair", "next_1d_ret"]].copy()
    work["position"] = pos.reindex(work.index).fillna(0.0).values
    if dates is not None:
        date_set = set(pd.to_datetime(dates))
        work = work[work["date"].isin(date_set)]
    date_values = sorted(work["date"].unique())
    pv = [INITIAL_CAPITAL]
    daily_returns: list[float] = []
    gross_exposures: list[float] = []
    long_exposures: list[float] = []
    short_exposures: list[float] = []
    trades = 0
    for date in date_values:
        day = work[work["date"] == date]
        weights = day["position"].values.astype(float)
        returns = day["next_1d_ret"].values.astype(float)
        gross = float(np.abs(weights).sum())
        pnl = float(np.sum(weights * returns))
        fee_cost = fee_rate * gross
        net = pnl - fee_cost
        pv.append(pv[-1] * (1.0 + net))
        daily_returns.append(net)
        gross_exposures.append(gross)
        long_exposures.append(float(np.clip(weights, 0.0, None).sum()))
        short_exposures.append(float(np.clip(-weights, 0.0, None).sum()))
        trades += int(np.count_nonzero(np.abs(weights) > 1e-12))

    pv_arr = np.array(pv, dtype=float)
    dr_arr = np.array(daily_returns, dtype=float)
    if len(dr_arr) == 0:
        sharpe = 0.0
        win_rate = 0.0
    else:
        sharpe = float(dr_arr.mean() / (dr_arr.std() + 1e-10) * math.sqrt(365))
        win_rate = float((dr_arr > 0).mean() * 100.0)
    peak = np.maximum.accumulate(pv_arr)
    mdd = float(((pv_arr - peak) / peak).min() * 100.0)
    total_ret = float((pv_arr[-1] / pv_arr[0] - 1.0) * 100.0)
    metrics = {
        "strategy": name,
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(win_rate, 1),
        "trades": int(trades),
        "final_value": round(float(pv_arr[-1]), 2),
        "avg_gross_exposure": round(float(np.mean(gross_exposures)) if gross_exposures else 0.0, 4),
        "avg_long_exposure": round(float(np.mean(long_exposures)) if long_exposures else 0.0, 4),
        "avg_short_exposure": round(float(np.mean(short_exposures)) if short_exposures else 0.0, 4),
        "score": round(rank_score(
            {
                "sharpe": sharpe,
                "total_return_pct": total_ret,
                "max_drawdown_pct": mdd,
                "trades": trades,
            }
        ), 4),
        "daily_returns": daily_returns,
        "equity_curve": pv,
    }
    return metrics


def buy_and_hold_metrics(df: pd.DataFrame, dates: list[pd.Timestamp], name: str = "Buy&Hold") -> dict[str, object]:
    work = df[df["date"].isin(set(dates))].copy()
    date_values = sorted(work["date"].unique())
    pv = [INITIAL_CAPITAL]
    daily_returns: list[float] = []
    for date in date_values:
        day = work[work["date"] == date]
        net = float(day["next_1d_ret"].mean()) if len(day) else 0.0
        pv.append(pv[-1] * (1.0 + net))
        daily_returns.append(net)
    pv_arr = np.array(pv, dtype=float)
    dr_arr = np.array(daily_returns, dtype=float)
    sharpe = float(dr_arr.mean() / (dr_arr.std() + 1e-10) * math.sqrt(365)) if len(dr_arr) else 0.0
    peak = np.maximum.accumulate(pv_arr)
    mdd = float(((pv_arr - peak) / peak).min() * 100.0)
    total_ret = float((pv_arr[-1] / pv_arr[0] - 1.0) * 100.0)
    return {
        "strategy": name,
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(float((dr_arr > 0).mean() * 100.0) if len(dr_arr) else 0.0, 1),
        "trades": int(len(date_values) * NUM_PAIRS),
        "final_value": round(float(pv_arr[-1]), 2),
        "avg_gross_exposure": 1.0,
        "avg_long_exposure": 1.0,
        "avg_short_exposure": 0.0,
        "score": round(rank_score(
            {
                "sharpe": sharpe,
                "total_return_pct": total_ret,
                "max_drawdown_pct": mdd,
                "trades": len(date_values) * NUM_PAIRS,
            }
        ), 4),
        "daily_returns": daily_returns,
        "equity_curve": pv,
    }


def positions_ls(df: pd.DataFrame, threshold: float, long_only: bool = False) -> pd.Series:
    pos = np.zeros(len(df), dtype=float)
    pred = df["pred_3d"].values
    pos[pred > threshold] = POS_SIZE
    if not long_only:
        pos[pred < -threshold] = -POS_SIZE
    return pd.Series(pos, index=df.index)


def positions_topk(df: pd.DataFrame, k: int) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    for _, day in df.groupby("date"):
        if len(day) < 2 * k:
            continue
        ordered = day.sort_values("pred_3d", ascending=False)
        weight = 1.0 / (2 * k)
        pos.loc[ordered.head(k).index] = weight
        pos.loc[ordered.tail(k).index] = -weight
    return pos


def positions_quantile(df: pd.DataFrame, q: float) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    for _, day in df.groupby("date"):
        n = len(day)
        k = max(1, int(round(n * q)))
        ordered = day.sort_values("pred_3d", ascending=False)
        weight = 1.0 / (2 * k)
        pos.loc[ordered.head(k).index] = weight
        pos.loc[ordered.tail(k).index] = -weight
    return pos


def positions_always(df: pd.DataFrame, side: str) -> pd.Series:
    if side == "short":
        return pd.Series(-POS_SIZE, index=df.index)
    if side == "long":
        return pd.Series(POS_SIZE, index=df.index)
    raise ValueError(side)


def positions_random(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(df))
    return pd.Series(signs * POS_SIZE, index=df.index)


def positions_vol_target(df: pd.DataFrame, threshold: float, q: float | None = None) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    vol = df["vol_20"].replace(0, np.nan).fillna(df["vol_20"].median()).values
    score = df["pred_3d"].values / np.maximum(vol, 1e-4)
    tmp = df[["date"]].copy()
    tmp["score"] = score
    tmp["pred_abs"] = df["pred_3d"].abs().values
    for date, day in tmp.groupby("date"):
        idx = day.index
        s = day["score"].values.astype(float)
        if q is not None:
            n = len(day)
            k = max(1, int(round(n * q)))
            ordered = day.sort_values("score", ascending=False)
            active_idx = list(ordered.head(k).index) + list(ordered.tail(k).index)
        else:
            active_idx = list(day[day["pred_abs"] > threshold].index)
        if not active_idx:
            continue
        active_scores = pd.Series(score[active_idx], index=active_idx).clip(-5.0, 5.0)
        denom = float(active_scores.abs().sum())
        if denom <= 1e-12:
            continue
        pos.loc[active_idx] = active_scores / denom
    return pos


def positions_regime_filter(df: pd.DataFrame, threshold: float, weak_scale: float = 0.25) -> pd.Series:
    base = positions_ls(df, threshold)
    pos = base.copy()
    bear = df["btc_mom_60"].values < 0
    bull = df["btc_mom_60"].values > 0
    long_mask = pos.values > 0
    short_mask = pos.values < 0
    values = pos.values.copy()
    values[bear & long_mask] *= weak_scale
    values[bull & short_mask] *= weak_scale
    return pd.Series(values, index=df.index)


def apply_drawdown_overlay(df: pd.DataFrame, base_pos: pd.Series, threshold: float = -0.12) -> pd.Series:
    pos = base_pos.copy().astype(float)
    scaled = pd.Series(0.0, index=df.index)
    pv = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    for date in sorted(df["date"].unique()):
        idx = df.index[df["date"] == date]
        dd = pv / peak - 1.0
        if dd < 2 * threshold:
            scale = 0.25
        elif dd < threshold:
            scale = 0.50
        else:
            scale = 1.0
        scaled.loc[idx] = pos.loc[idx] * scale
        day = df.loc[idx]
        net = float((scaled.loc[idx].values * day["next_1d_ret"].values).sum())
        net -= FEE_RATE * float(np.abs(scaled.loc[idx].values).sum())
        pv *= 1.0 + net
        peak = max(peak, pv)
    return scaled


def build_rule_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    print("\n[5/7] Building baseline/rule candidate strategies...")
    candidates: dict[str, pd.Series] = {}
    for th in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]:
        candidates[f"V4-LS(th={th:.4f})"] = positions_ls(df, th)
    for th in [0.0, 0.005, 0.01, 0.015]:
        candidates[f"V4-LO(th={th:.4f})"] = positions_ls(df, th, long_only=True)
    for k in [1, 2, 3, 5]:
        candidates[f"Top{k}-LS"] = positions_topk(df, k)
    for q in [0.10, 0.20, 0.30, 0.40]:
        candidates[f"Quantile-LS(q={q:.2f})"] = positions_quantile(df, q)
    for th in [0.0, 0.005, 0.01]:
        candidates[f"VolTarget(th={th:.3f})"] = positions_vol_target(df, th, q=None)
    for q in [0.10, 0.20, 0.30]:
        candidates[f"VolTarget-Quantile(q={q:.2f})"] = positions_vol_target(df, 0.0, q=q)
    for th in [0.0, 0.005, 0.01]:
        candidates[f"RegimeFilter(th={th:.3f})"] = positions_regime_filter(df, th, weak_scale=0.25)
    candidates["AlwaysShort(daily_fee)"] = positions_always(df, "short")
    candidates["AlwaysLong(daily_fee)"] = positions_always(df, "long")
    candidates["DrawdownOverlay(V4-LS0.005)"] = apply_drawdown_overlay(df, positions_ls(df, 0.005))
    candidates["DrawdownOverlay(Regime0.005)"] = apply_drawdown_overlay(
        df, positions_regime_filter(df, 0.005)
    )
    print(f"  candidates={len(candidates)}")
    return candidates


def run_walkforward_rule_optimizer(
    df: pd.DataFrame,
    candidates: dict[str, pd.Series],
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]],
) -> tuple[pd.Series, dict[str, int], list[dict[str, object]]]:
    print("  Running walk-forward rule selector...")
    combined = pd.Series(0.0, index=df.index)
    selection_counts: Counter[str] = Counter()
    fold_details: list[dict[str, object]] = []
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        best_name = ""
        best_score = -1e9
        best_metrics: dict[str, object] | None = None
        for name, pos in candidates.items():
            m = simulate_positions(df, pos, name, dates=train_dates)
            score = float(m["score"])
            if score > best_score:
                best_name = name
                best_score = score
                best_metrics = m
        assert best_metrics is not None
        test_idx = df.index[df["date"].isin(set(test_dates))]
        combined.loc[test_idx] = candidates[best_name].loc[test_idx]
        selection_counts[best_name] += 1
        fold_details.append(
            {
                "fold": fold_id,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "selected": best_name,
                "train_score": best_score,
                "train_return_pct": best_metrics["total_return_pct"],
                "train_sharpe": best_metrics["sharpe"],
            }
        )
        print(
            f"    fold {fold_id}: {test_dates[0].date()}~{test_dates[-1].date()} "
            f"selected={best_name} train_score={best_score:+.3f}"
        )
    return combined, dict(selection_counts), fold_details


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def scale_fold_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
    x_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num((x_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train.astype(np.float32), x_test.astype(np.float32)


def maybe_parallel(model: nn.Module, min_batch: int) -> nn.Module:
    if torch.cuda.device_count() > 1 and min_batch >= torch.cuda.device_count():
        return nn.DataParallel(model)
    return model


def train_meta_label_positions(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]],
    device: torch.device,
) -> dict[str, pd.Series]:
    print("\n[6/7] Training GPU meta-labeling models walk-forward...")
    set_seed(9001)
    probs = pd.Series(0.0, index=df.index)
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train_mask = df["date"].isin(set(train_dates)).values
        test_mask = df["date"].isin(set(test_dates)).values
        x_train, x_test = scale_fold_features(df, feature_cols, train_mask, test_mask)
        signed_ret = np.sign(df.loc[train_mask, "pred_3d"].values) * df.loc[train_mask, "next_1d_ret"].values
        y_train = (signed_ret > FEE_RATE).astype(np.float32)
        if y_train.sum() < 5 or len(y_train) - y_train.sum() < 5:
            continue
        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
        pos_weight = float((len(y_train) - y_train.sum()) / max(1.0, y_train.sum()))
        model = maybe_parallel(MLP(len(feature_cols), 1, hidden=96, dropout=0.20).to(device), len(x_train))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        for epoch in range(META_EPOCHS):
            model.train()
            opt.zero_grad()
            logits = model(x_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            x_eval = torch.tensor(x_test, dtype=torch.float32, device=device)
            p = torch.sigmoid(model(x_eval)).detach().cpu().numpy().reshape(-1)
        probs.loc[df.index[test_mask]] = p
        print(
            f"    meta fold {fold_id}: {test_dates[0].date()}~{test_dates[-1].date()} "
            f"train_pos={y_train.mean():.3f} prob_mean={p.mean():.3f}"
        )

    outputs: dict[str, pd.Series] = {}
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        signs = np.sign(df["pred_3d"].values)
        pos = np.where(probs.values >= threshold, signs * POS_SIZE, 0.0)
        outputs[f"MetaLabel(th={threshold:.2f})"] = pd.Series(pos, index=df.index)
    return outputs


def train_bandit_positions(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]],
    device: torch.device,
) -> dict[str, pd.Series]:
    print("\n[6/7] Training GPU contextual bandit Q-models walk-forward...")
    set_seed(9101)
    free_actions = pd.Series(0.0, index=df.index)
    masked_actions = pd.Series(0.0, index=df.index)
    actions_t = torch.tensor(ACTIONS.reshape(1, -1), dtype=torch.float32, device=device)
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train_mask = df["date"].isin(set(train_dates)).values
        test_mask = df["date"].isin(set(test_dates)).values
        x_train, x_test = scale_fold_features(df, feature_cols, train_mask, test_mask)
        ret = df.loc[train_mask, "next_1d_ret"].values.astype(np.float32).reshape(-1, 1)
        y_train = ret * ACTIONS.reshape(1, -1) - FEE_RATE * np.abs(ACTIONS.reshape(1, -1))
        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        model = maybe_parallel(MLP(len(feature_cols), len(ACTIONS), hidden=96, dropout=0.15).to(device), len(x_train))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        loss_fn = nn.SmoothL1Loss(beta=0.005)
        for _ in range(BANDIT_EPOCHS):
            model.train()
            opt.zero_grad()
            pred_q = model(x_t)
            loss = loss_fn(pred_q, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            x_eval = torch.tensor(x_test, dtype=torch.float32, device=device)
            q = model(x_eval).detach().cpu().numpy()

        free_idx = q.argmax(axis=1)
        free_action_values = ACTIONS[free_idx]
        test_pred_sign = np.sign(df.loc[test_mask, "pred_3d"].values.astype(float))
        masked_action_values = np.zeros(len(q), dtype=np.float32)
        for i, sign in enumerate(test_pred_sign):
            if sign > 0:
                valid = np.array([2, 3, 4])
            elif sign < 0:
                valid = np.array([0, 1, 2])
            else:
                valid = np.array([2])
            best_local = valid[q[i, valid].argmax()]
            masked_action_values[i] = ACTIONS[best_local]

        free_actions.loc[df.index[test_mask]] = free_action_values / NUM_PAIRS
        masked_actions.loc[df.index[test_mask]] = masked_action_values / NUM_PAIRS
        print(
            f"    bandit fold {fold_id}: {test_dates[0].date()}~{test_dates[-1].date()} "
            f"free_mean={free_action_values.mean():+.3f} masked_mean={masked_action_values.mean():+.3f}"
        )
    return {
        "Bandit-FreeQ": free_actions,
        "Bandit-V4MaskedQ": masked_actions,
    }


def policy_daily_return(
    model: nn.Module,
    x: torch.Tensor,
    r: torch.Tensor,
    groups: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    daily_returns = []
    gross_values = []
    for idx in groups:
        raw = torch.tanh(model(x[idx]).squeeze(-1))
        denom = torch.clamp(raw.abs().sum(), min=1.0)
        weights = raw / denom
        daily = (weights * r[idx]).sum() - FEE_RATE * weights.abs().sum()
        daily_returns.append(daily)
        gross_values.append(weights.abs().sum())
    return torch.stack(daily_returns), torch.stack(gross_values)


def train_policy_gradient_positions(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]],
    device: torch.device,
) -> dict[str, pd.Series]:
    print("\n[6/7] Training GPU direct policy-gradient allocator walk-forward...")
    set_seed(9201)
    out_pos = pd.Series(0.0, index=df.index)
    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train_mask = df["date"].isin(set(train_dates)).values
        test_mask = df["date"].isin(set(test_dates)).values
        x_train, x_test = scale_fold_features(df, feature_cols, train_mask, test_mask)
        train_frame = df.loc[train_mask, ["date", "next_1d_ret"]].reset_index(drop=True)
        test_frame = df.loc[test_mask, ["date", "next_1d_ret"]].reset_index()
        train_groups = [
            torch.tensor(v, dtype=torch.long, device=device)
            for v in train_frame.groupby("date").indices.values()
        ]
        x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
        r_t = torch.tensor(train_frame["next_1d_ret"].values.astype(np.float32), device=device)
        model = MLP(len(feature_cols), 1, hidden=96, dropout=0.10).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
        for epoch in range(POLICY_EPOCHS):
            model.train()
            opt.zero_grad()
            daily_ret, gross = policy_daily_return(model, x_t, r_t, train_groups)
            mean = daily_ret.mean()
            std = daily_ret.std(unbiased=False) + 1e-5
            sharpe = mean / std * math.sqrt(365)
            clipped = torch.clamp(daily_ret, min=-0.50, max=0.50)
            pv = torch.cumprod(1.0 + clipped, dim=0)
            peak = torch.cummax(pv, dim=0).values
            drawdown = (pv - peak) / peak
            dd_penalty = torch.relu((-drawdown.min()) - 0.25)
            exposure_penalty = torch.relu(gross.mean() - 0.80)
            loss = -sharpe - 0.15 * mean * 365 + 0.60 * dd_penalty + 0.05 * exposure_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        x_eval = torch.tensor(x_test, dtype=torch.float32, device=device)
        positions = np.zeros(len(test_frame), dtype=np.float32)
        with torch.no_grad():
            for _, idx_values in test_frame.groupby("date").indices.items():
                idx_np = np.array(idx_values, dtype=np.int64)
                idx_t = torch.tensor(idx_np, dtype=torch.long, device=device)
                raw = torch.tanh(model(x_eval[idx_t]).squeeze(-1))
                denom = torch.clamp(raw.abs().sum(), min=1.0)
                weights = (raw / denom).detach().cpu().numpy()
                positions[idx_np] = weights.astype(np.float32)
        out_pos.loc[test_frame["index"].values] = positions
        print(
            f"    policy fold {fold_id}: {test_dates[0].date()}~{test_dates[-1].date()} "
            f"train_days={len(train_groups)} avg_abs_pos={np.mean(np.abs(positions)):.4f}"
        )
    return {"PolicyGradient-Allocator": out_pos}


def evaluate_all_strategies(
    df: pd.DataFrame,
    position_map: dict[str, pd.Series],
    eval_dates: list[pd.Timestamp],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, pos in position_map.items():
        metrics = simulate_positions(df, pos, name, dates=eval_dates)
        results[name] = metrics
    results["Buy&Hold"] = buy_and_hold_metrics(df, eval_dates)
    random_metrics = []
    for seed in range(30):
        m = simulate_positions(df, positions_random(df, 10_000 + seed), f"RandomLS-{seed}", dates=eval_dates)
        random_metrics.append(m)
    rand_ret = float(np.mean([m["total_return_pct"] for m in random_metrics]))
    rand_sharpe = float(np.mean([m["sharpe"] for m in random_metrics]))
    rand_mdd = float(np.mean([m["max_drawdown_pct"] for m in random_metrics]))
    results["RandomLS(mean30)"] = {
        "strategy": "RandomLS(mean30)",
        "total_return_pct": round(rand_ret, 2),
        "sharpe": round(rand_sharpe, 2),
        "max_drawdown_pct": round(rand_mdd, 2),
        "win_rate_pct": round(float(np.mean([m["win_rate_pct"] for m in random_metrics])), 1),
        "trades": int(np.mean([m["trades"] for m in random_metrics])),
        "final_value": round(float(np.mean([m["final_value"] for m in random_metrics])), 2),
        "avg_gross_exposure": 1.0,
        "avg_long_exposure": 0.5,
        "avg_short_exposure": 0.5,
        "score": round(float(np.mean([m["score"] for m in random_metrics])), 4),
        "daily_returns": [],
        "equity_curve": [],
    }
    return results


def summarize_prediction_quality(df: pd.DataFrame, eval_dates: list[pd.Timestamp]) -> dict[str, object]:
    work = df[df["date"].isin(set(eval_dates))].copy()
    pred = work["pred_3d"].values
    true_3d = work["actual_3d_ret"].values
    true_1d = work["next_1d_ret"].values
    ic_3d = float(np.corrcoef(pred, true_3d)[0, 1])
    ic_1d = float(np.corrcoef(pred, true_1d)[0, 1])
    dir_3d = float(((pred > 0) == (true_3d > 0)).mean())
    dir_1d = float(((pred > 0) == (true_1d > 0)).mean())
    return {
        "eval_rows": int(len(work)),
        "eval_days": int(work["date"].nunique()),
        "period": f"{work['date'].min().date()} to {work['date'].max().date()}",
        "ic_3d": ic_3d,
        "ic_1d": ic_1d,
        "dir_acc_3d": dir_3d,
        "dir_acc_1d": dir_1d,
        "mean_next_1d_ret": float(true_1d.mean()),
        "mean_actual_3d_ret": float(true_3d.mean()),
    }


def clean_results_for_json(results: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    cleaned: dict[str, dict[str, object]] = {}
    for name, metrics in results.items():
        cleaned[name] = {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}
    return cleaned


def plot_outputs(results: dict[str, dict[str, object]], selection_counts: dict[str, int]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        [m for m in results.values() if m.get("equity_curve")],
        key=lambda x: float(x.get("score", -999)),
        reverse=True,
    )
    keep_names = {m["strategy"] for m in ranked[:8]}
    for extra in ["V4-LS(th=0.0050)", "AlwaysShort(daily_fee)", "Buy&Hold"]:
        if extra in results:
            keep_names.add(extra)

    plt.figure(figsize=(14, 8))
    for metrics in results.values():
        name = str(metrics["strategy"])
        curve = metrics.get("equity_curve")
        if name in keep_names and curve:
            plt.plot(curve, label=name, linewidth=1.6)
    plt.title("V9 Strategy Equity Curves (walk-forward evaluation window)")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Trading Days")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v9_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    table = sorted(results.values(), key=lambda x: float(x.get("score", -999)), reverse=True)[:16]
    names = [str(m["strategy"]) for m in table]
    rets = [float(m["total_return_pct"]) for m in table]
    sharpes = [float(m["sharpe"]) for m in table]
    mdds = [float(m["max_drawdown_pct"]) for m in table]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].barh(names[::-1], rets[::-1], color="steelblue")
    axes[0].set_title("Total Return %")
    axes[1].barh(names[::-1], sharpes[::-1], color="seagreen")
    axes[1].set_title("Sharpe")
    axes[2].barh(names[::-1], mdds[::-1], color="indianred")
    axes[2].set_title("Max Drawdown %")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v9_strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    if selection_counts:
        labels = list(selection_counts.keys())
        values = [selection_counts[k] for k in labels]
        plt.figure(figsize=(12, 6))
        plt.barh(labels[::-1], values[::-1], color="darkorange")
        plt.title("Walk-forward Rule Optimizer Selection Counts")
        plt.xlabel("Folds selected")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "v9_rule_selection_counts.png", dpi=150, bbox_inches="tight")
        plt.close()


def write_report(
    results: dict[str, dict[str, object]],
    prediction_quality: dict[str, object],
    selection_counts: dict[str, int],
    fold_details: list[dict[str, object]],
) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results.values(), key=lambda x: float(x.get("score", -999)), reverse=True)
    best = ranked[0]
    adaptive = [
        m for m in ranked
        if any(tag in str(m["strategy"]) for tag in ["Meta", "Bandit", "Policy", "WalkForward", "Regime", "VolTarget", "Drawdown"])
    ]
    best_adaptive = adaptive[0] if adaptive else best
    base_v4 = results.get("V4-LS(th=0.0050)", {})
    always_short = results.get("AlwaysShort(daily_fee)", {})

    def row(metrics: dict[str, object]) -> str:
        return (
            f"| {metrics['strategy']} | {metrics['total_return_pct']:+.2f}% | "
            f"{metrics['sharpe']:.2f} | {metrics['max_drawdown_pct']:.2f}% | "
            f"{metrics['win_rate_pct']:.1f}% | {metrics['trades']} | {metrics.get('score', 0):+.3f} |"
        )

    lines: list[str] = []
    lines.append("# V9：V4 信号交易层优化 + Meta/Bandit/RL 实验报告")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append(
        "V9 不再尝试重新预测价格，而是把 V4 的弱预测信号当作 alpha 输入，系统测试多种交易层优化方法："
        "固定阈值、Top-K、动态分位数、波动率目标、BTC regime filter、回撤 overlay、walk-forward 规则选择、"
        "meta-labeling、contextual bandit，以及直接 policy-gradient 仓位分配。"
    )
    lines.append("")
    lines.append("## 2. 数据与验证方式")
    lines.append("")
    lines.append(f"- V4 checkpoint：只加载 `data/checkpoints/v4_seed_*.pt`，不重新训练 V4。")
    lines.append(f"- 评估区间：`{prediction_quality['period']}`。")
    lines.append(f"- 样本：{prediction_quality['eval_rows']} 条预测，{prediction_quality['eval_days']} 个交易日。")
    lines.append(f"- 手续费：单边 `fee_rate={FEE_RATE}`，按每日实际暴露收取。")
    lines.append(f"- 训练方式：meta/bandit/policy 都采用 expanding walk-forward，只用过去窗口训练，未来 30 天交易。")
    lines.append(f"- 排名分数：`Sharpe + Return - DrawdownPenalty - TradePenalty`，避免只按收益率选过拟合策略。")
    lines.append("")
    lines.append("V4 信号在本评估窗口内的预测质量：")
    lines.append("")
    lines.append(f"- 3日 IC：{prediction_quality['ic_3d']:+.4f}")
    lines.append(f"- 1日 IC：{prediction_quality['ic_1d']:+.4f}")
    lines.append(f"- 3日方向准确率：{prediction_quality['dir_acc_3d']:.2%}")
    lines.append(f"- 1日方向准确率：{prediction_quality['dir_acc_1d']:.2%}")
    lines.append("")
    lines.append("## 3. 主要结果")
    lines.append("")
    lines.append("| 策略 | 收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | Score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metrics in ranked[:20]:
        lines.append(row(metrics))
    lines.append("")
    lines.append("## 4. 最优策略判断")
    lines.append("")
    lines.append(
        f"按风险调整分数，本次 V9 的全体最优是 **{best['strategy']}**："
        f"收益 {best['total_return_pct']:+.2f}%，Sharpe {best['sharpe']:.2f}，"
        f"最大回撤 {best['max_drawdown_pct']:.2f}%。"
    )
    lines.append("")
    lines.append(
        f"如果只看真正的优化/学习类策略，最优是 **{best_adaptive['strategy']}**："
        f"收益 {best_adaptive['total_return_pct']:+.2f}%，Sharpe {best_adaptive['sharpe']:.2f}，"
        f"最大回撤 {best_adaptive['max_drawdown_pct']:.2f}%。"
    )
    lines.append("")
    if base_v4:
        lines.append(
            f"参考 V4 固定阈值 `LS(th=0.005)` 在同一评估窗口下：收益 {base_v4['total_return_pct']:+.2f}%，"
            f"Sharpe {base_v4['sharpe']:.2f}，最大回撤 {base_v4['max_drawdown_pct']:.2f}%。"
        )
    if always_short:
        lines.append(
            f"关键基线 `AlwaysShort(daily_fee)`：收益 {always_short['total_return_pct']:+.2f}%，"
            f"Sharpe {always_short['sharpe']:.2f}，最大回撤 {always_short['max_drawdown_pct']:.2f}%。"
            "如果它排名很高，说明市场 regime 本身贡献很大，不能把收益完全归功于模型。"
        )
    lines.append("")
    lines.append("## 5. Walk-forward 规则选择器")
    lines.append("")
    if selection_counts:
        lines.append("规则选择次数：")
        lines.append("")
        for name, count in sorted(selection_counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- `{name}`：{count} 次")
        lines.append("")
    lines.append("各 fold：")
    lines.append("")
    lines.append("| Fold | 训练区间 | 交易区间 | 选择策略 | 训练收益 | 训练Sharpe |")
    lines.append("|---:|---|---|---|---:|---:|")
    for fd in fold_details:
        lines.append(
            f"| {fd['fold']} | {fd['train_start']}~{fd['train_end']} | "
            f"{fd['test_start']}~{fd['test_end']} | {fd['selected']} | "
            f"{fd['train_return_pct']}% | {fd['train_sharpe']} |"
        )
    lines.append("")
    lines.append("## 6. 图表")
    lines.append("")
    lines.append("- `docs/figures/v9_equity_curves.png`：主要策略净值曲线。")
    lines.append("- `docs/figures/v9_strategy_comparison.png`：收益、Sharpe、回撤对比。")
    lines.append("- `docs/figures/v9_rule_selection_counts.png`：walk-forward 规则选择次数。")
    lines.append("")
    lines.append("## 7. 结论")
    lines.append("")
    lines.append(
        "V9 的核心判断不是“RL 能不能预测市场”，而是“在 V4 弱信号之上，哪种交易层能最稳健地把信号转化为收益”。"
        "如果 meta/bandit/policy 没有显著超过简单规则或 always-short，说明当前日频数据里的可泛化 alpha 仍然不足；"
        "如果某个 walk-forward 优化策略超过 V4 且回撤更低，它才是当前最值得继续扩展的方向。"
    )
    lines.append("")
    (DOC_DIR / "v9_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(2026)
    device = require_cuda()
    pred_df = get_or_create_predictions(device)
    df, feature_cols = add_state_features(pred_df)
    dates = sorted(pd.to_datetime(df["date"].unique()))
    if len(dates) <= MIN_TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough dates for V9 walk-forward: {len(dates)}")
    folds = fold_slices(dates)
    eval_dates = dates[MIN_TRAIN_DAYS:]
    print(
        f"  walk-forward folds={len(folds)}, "
        f"eval_period={eval_dates[0].date()} to {eval_dates[-1].date()}"
    )

    candidates = build_rule_candidates(df)
    wf_pos, selection_counts, fold_details = run_walkforward_rule_optimizer(df, candidates, folds)

    position_map: dict[str, pd.Series] = dict(candidates)
    position_map["WalkForwardRuleSelector"] = wf_pos
    position_map.update(train_meta_label_positions(df, feature_cols, folds, device))
    position_map.update(train_bandit_positions(df, feature_cols, folds, device))
    position_map.update(train_policy_gradient_positions(df, feature_cols, folds, device))

    print("\n[7/7] Evaluating all strategies and writing outputs...")
    results = evaluate_all_strategies(df, position_map, eval_dates)
    prediction_quality = summarize_prediction_quality(df, eval_dates)
    cleaned = clean_results_for_json(results)
    ranked = sorted(cleaned.values(), key=lambda x: float(x.get("score", -999)), reverse=True)
    output = {
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "fee_rate": FEE_RATE,
            "min_train_days": MIN_TRAIN_DAYS,
            "trade_window_days": TRADE_WINDOW_DAYS,
            "meta_epochs": META_EPOCHS,
            "bandit_epochs": BANDIT_EPOCHS,
            "policy_epochs": POLICY_EPOCHS,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_version": torch.__version__,
            "v4_prediction_file": str(PRED_PARQUET.relative_to(ROOT)),
        },
        "prediction_quality": prediction_quality,
        "strategies": cleaned,
        "ranked_strategies": ranked,
        "best_strategy": ranked[0],
        "walkforward_rule_selection_counts": selection_counts,
        "walkforward_rule_folds": fold_details,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "backtest_v9_results.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    with (DATA_DIR / "results_v9.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_strategy": ranked[0],
                "prediction_quality": prediction_quality,
                "top10": ranked[:10],
                "config": output["config"],
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    plot_outputs(results, selection_counts)
    write_report(results, prediction_quality, selection_counts, fold_details)

    print("\nTop strategies:")
    for metrics in ranked[:12]:
        print(
            f"  {metrics['strategy']:32s} ret={metrics['total_return_pct']:+7.2f}% "
            f"sharpe={metrics['sharpe']:6.2f} mdd={metrics['max_drawdown_pct']:7.2f}% "
            f"score={metrics['score']:+.3f} trades={metrics['trades']}"
        )
    print("\nSaved:")
    print("  data/backtest_v9_results.json")
    print("  data/results_v9.json")
    print("  docs/v9_experiment_report.md")
    print("  docs/figures/v9_equity_curves.png")
    print("  docs/figures/v9_strategy_comparison.png")
    print("  docs/figures/v9_rule_selection_counts.png")


if __name__ == "__main__":
    main()
