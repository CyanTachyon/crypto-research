"""V12: 4h high-frequency crypto trading experiment.

This script intentionally does not reuse daily V4/V9/V10 prediction artifacts.
It loads 15-pair Binance 4h OHLCV cache, builds bar-based features, trains a
cost-aware Transformer edge model with purged walk-forward splits, compares
against simple 4h baselines, and writes Chinese reports/results.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
    "UNI/USDT", "AAVE/USDT", "ATOM/USDT", "NEAR/USDT", "OP/USDT",
]

TIMEFRAME = "4h"
BAR_PER_DAY = 6
ANNUALIZATION = math.sqrt(365 * BAR_PER_DAY)
SEQ_LEN = 120
LABEL_HORIZON = 6
MAX_HORIZON = 12
TRAIN_EPOCHS = 12
BATCH_SIZE = 768
FIT_LR = 2e-4
WEIGHT_DECAY = 2e-3
INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_DAILY = 0.0001
FUNDING_PER_BAR = FUNDING_DAILY / BAR_PER_DAY
MIN_TRAIN_BARS = 1800
TRADE_WINDOW_BARS = 1440
PURGE_BARS = SEQ_LEN + MAX_HORIZON
SEED = 20260608

RESULTS_PATH = DATA_DIR / "results_v12.json"
BACKTEST_PATH = DATA_DIR / "backtest_v12_results.json"
DECISIONS_PATH = DATA_DIR / "v12_trade_decisions.parquet"
REPORT_PATH = DOCS_DIR / "v12_experiment_report.md"

FEATURE_COLS = [
    "log_ret_1", "hl_range", "oc_ret", "volume_log", "volume_ratio_24",
    "ret_3", "ret_6", "ret_12", "ret_24", "ret_42", "ret_72",
    "vol_6", "vol_12", "vol_24", "vol_42", "vol_72",
    "range_6", "range_24", "price_pos_24", "price_pos_72",
    "mom_z_12", "mom_z_24", "vol_ratio_6_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "ret_rank_6", "ret_z_cs", "vol_rank_24",
    "btc_ret_6", "btc_ret_24", "btc_vol_24",
    "market_ret_6", "market_vol_24",
]


def safe_pair(pair: str) -> str:
    return pair.replace("/", "_")


def raw_path(pair: str, timeframe: str = TIMEFRAME) -> Path:
    return RAW_DIR / f"binance_{safe_pair(pair)}_{timeframe}.parquet"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V12 training; CPU fallback is forbidden")
    for idx in range(torch.cuda.device_count()):
        print(f"  cuda:{idx} {torch.cuda.get_device_name(idx)}", flush=True)
    return torch.device("cuda")


def load_4h_cache() -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for pair in PAIRS:
        path = raw_path(pair)
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_parquet(path).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df) < MIN_TRAIN_BARS + SEQ_LEN + TRADE_WINDOW_BARS:
            raise RuntimeError(f"Insufficient 4h rows for {pair}: {len(df)}")
        raw[pair] = df
    if missing:
        raise FileNotFoundError("Missing 4h cache files:\n" + "\n".join(missing))
    return raw


def add_pair_features(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["pair"] = pair
    out["pair_id"] = PAIRS.index(pair)
    out["log_ret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_ret"] = out["close"] / out["open"] - 1.0
    out["volume_log"] = np.log1p(out["volume"])
    out["volume_ratio_24"] = out["volume"] / out["volume"].rolling(24).mean()
    for w in [3, 6, 12, 24, 42, 72]:
        out[f"ret_{w}"] = np.log(out["close"] / out["close"].shift(w))
    for w in [6, 12, 24, 42, 72]:
        out[f"vol_{w}"] = out["log_ret_1"].rolling(w).std()
    for w in [6, 24]:
        out[f"range_{w}"] = (out["high"].rolling(w).max() - out["low"].rolling(w).min()) / out["close"]
    for w in [24, 72]:
        low = out["low"].rolling(w).min()
        high = out["high"].rolling(w).max()
        out[f"price_pos_{w}"] = (out["close"] - low) / (high - low).replace(0, np.nan)
    out["mom_z_12"] = out["ret_12"] / out["vol_12"].replace(0, np.nan)
    out["mom_z_24"] = out["ret_24"] / out["vol_24"].replace(0, np.nan)
    out["vol_ratio_6_24"] = out["vol_6"] / out["vol_24"].replace(0, np.nan)
    hour = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["next_1bar_ret"] = out["close"].shift(-1) / out["close"] - 1.0
    for h in [3, 6, 12]:
        out[f"fwd_{h}bar_ret"] = out["close"].shift(-h) / out["close"] - 1.0
    return out


def build_feature_frame(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [add_pair_features(pair, df) for pair, df in raw.items()]
    all_df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    all_df["ret_rank_6"] = all_df.groupby("timestamp")["ret_6"].rank(pct=True)
    ret_mean = all_df.groupby("timestamp")["ret_6"].transform("mean")
    ret_std = all_df.groupby("timestamp")["ret_6"].transform("std").replace(0, np.nan)
    all_df["ret_z_cs"] = (all_df["ret_6"] - ret_mean) / ret_std
    all_df["vol_rank_24"] = all_df.groupby("timestamp")["vol_24"].rank(pct=True)

    btc = all_df[all_df["pair"] == "BTC/USDT"][["timestamp", "ret_6", "ret_24", "vol_24"]].copy()
    btc = btc.rename(columns={"ret_6": "btc_ret_6", "ret_24": "btc_ret_24", "vol_24": "btc_vol_24"})
    all_df = all_df.merge(btc, on="timestamp", how="left")
    all_df["market_ret_6"] = all_df.groupby("timestamp")["ret_6"].transform("mean")
    all_df["market_vol_24"] = all_df.groupby("timestamp")["log_ret_1"].transform("std")

    threshold = (FEE_RATE + SLIPPAGE_RATE) * 2.0 + FUNDING_PER_BAR * LABEL_HORIZON
    all_df["target_ret_6bar"] = all_df["fwd_6bar_ret"]
    all_df["target_z_6bar"] = all_df["target_ret_6bar"] / all_df["vol_24"].replace(0, np.nan)
    all_df["label_action"] = 1
    all_df.loc[all_df["target_ret_6bar"] > threshold, "label_action"] = 2
    all_df.loc[all_df["target_ret_6bar"] < -threshold, "label_action"] = 0

    keep = ["timestamp", "pair", "pair_id", "close", "next_1bar_ret", "fwd_3bar_ret", "fwd_6bar_ret", "fwd_12bar_ret", "target_ret_6bar", "target_z_6bar", "label_action"] + FEATURE_COLS
    all_df = all_df[keep].replace([np.inf, -np.inf], np.nan)
    all_df = all_df.dropna(subset=FEATURE_COLS + ["next_1bar_ret", "target_ret_6bar", "target_z_6bar"]).reset_index(drop=True)
    all_df["row_id"] = np.arange(len(all_df))
    return all_df


@dataclass(frozen=True)
class Fold:
    fold: int
    fit_dates: list[pd.Timestamp]
    select_dates: list[pd.Timestamp]
    trade_dates: list[pd.Timestamp]


def make_folds(dates: list[pd.Timestamp]) -> list[Fold]:
    folds: list[Fold] = []
    start = MIN_TRAIN_BARS
    fold_id = 1
    while start < len(dates):
        end = min(start + TRADE_WINDOW_BARS, len(dates))
        train_end = max(0, start - PURGE_BARS)
        train_dates = dates[:train_end]
        if len(train_dates) < MIN_TRAIN_BARS - PURGE_BARS:
            start = end
            continue
        split = max(1, int(len(train_dates) * 0.78))
        fit_dates = train_dates[:split]
        select_dates = train_dates[split:]
        if len(select_dates) < 120:
            select_dates = train_dates[-120:]
            fit_dates = train_dates[:-120]
        trade_dates = dates[start:end]
        if fit_dates and select_dates and trade_dates:
            folds.append(Fold(fold_id, fit_dates, select_dates, trade_dates))
            fold_id += 1
        start = end
    return folds


def fit_scaler(df: pd.DataFrame, fit_dates: list[pd.Timestamp]) -> tuple[pd.Series, pd.Series]:
    fit = df[df["timestamp"].isin(fit_dates)]
    mean = fit[FEATURE_COLS].mean()
    std = fit[FEATURE_COLS].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def build_sequences(
    df: pd.DataFrame,
    dates: list[pd.Timestamp],
    mean: pd.Series,
    std: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    date_set = set(dates)
    xs: list[np.ndarray] = []
    y_action: list[int] = []
    y_ret: list[float] = []
    row_ids: list[int] = []
    for pair in PAIRS:
        part = df[df["pair"] == pair].sort_values("timestamp").reset_index(drop=True).copy()
        feat = ((part[FEATURE_COLS] - mean) / std).clip(-8.0, 8.0).to_numpy(dtype=np.float32)
        timestamps = part["timestamp"].to_numpy()
        for idx in range(SEQ_LEN, len(part)):
            ts = pd.Timestamp(timestamps[idx])
            if ts not in date_set:
                continue
            xs.append(feat[idx - SEQ_LEN:idx])
            y_action.append(int(part.at[idx, "label_action"]))
            y_ret.append(float(np.clip(part.at[idx, "target_z_6bar"], -5.0, 5.0)))
            row_ids.append(int(part.at[idx, "row_id"]))
    if not xs:
        raise RuntimeError("No sequence samples built")
    return (
        torch.tensor(np.stack(xs), dtype=torch.float32),
        torch.tensor(y_action, dtype=torch.long),
        torch.tensor(y_ret, dtype=torch.float32),
        np.array(row_ids, dtype=np.int64),
    )


class V12Transformer(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        d_model = 96
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=256,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(d_model)
        self.action = nn.Linear(d_model, 3)
        self.ret = nn.Linear(d_model, 1)
        self.conf = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(z)
        h = self.norm(z[:, -1])
        return self.action(h), self.ret(h).squeeze(-1), torch.sigmoid(self.conf(h)).squeeze(-1)


def train_model(x: torch.Tensor, y_action: torch.Tensor, y_ret: torch.Tensor, device: torch.device, fold: int) -> nn.Module:
    model: nn.Module = V12Transformer(len(FEATURE_COLS)).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loader = DataLoader(TensorDataset(x, y_action, y_ret), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=FIT_LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()
    huber = nn.HuberLoss(delta=1.0)
    model.train()
    for epoch in range(1, TRAIN_EPOCHS + 1):
        total = 0.0
        for xb, ya, yr in loader:
            xb = xb.to(device, non_blocking=True)
            ya = ya.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, pred_ret, conf = model(xb)
            probs = torch.softmax(logits, dim=1)
            edge = probs[:, 2] - probs[:, 0]
            loss = ce(logits, ya) + 0.8 * huber(pred_ret, yr) + 0.08 * torch.mean((conf - torch.abs(edge).detach()).pow(2))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
        if epoch in {1, 10, 25, TRAIN_EPOCHS}:
            print(f"fold {fold} epoch {epoch:02d}/{TRAIN_EPOCHS} loss={total / len(x):.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, row_ids: np.ndarray, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    p_short: list[np.ndarray] = []
    p_flat: list[np.ndarray] = []
    p_long: list[np.ndarray] = []
    pred_ret: list[np.ndarray] = []
    conf: list[np.ndarray] = []
    model.eval()
    for (xb,) in loader:
        logits, ret, c = model(xb.to(device, non_blocking=True))
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        p_short.append(prob[:, 0])
        p_flat.append(prob[:, 1])
        p_long.append(prob[:, 2])
        pred_ret.append(ret.detach().cpu().numpy())
        conf.append(c.detach().cpu().numpy())
    out = pd.DataFrame({
        "row_id": row_ids,
        "p_short": np.concatenate(p_short),
        "p_flat": np.concatenate(p_flat),
        "p_long": np.concatenate(p_long),
        "pred_z": np.concatenate(pred_ret),
        "raw_conf": np.concatenate(conf),
    })
    out["edge"] = out["p_long"] - out["p_short"]
    out["confidence"] = np.maximum(out["p_long"], out["p_short"]) * (1.0 - out["p_flat"]) * out["raw_conf"]
    return out


def normalize_daily_gross(pos: pd.Series, df: pd.DataFrame, max_gross: float = 1.0) -> pd.Series:
    work = pd.DataFrame({"timestamp": df["timestamp"].values, "pos": pos.values}, index=df.index)
    gross = work.groupby("timestamp")["pos"].transform(lambda s: float(np.abs(s).sum()))
    gross_arr = gross.to_numpy(dtype=float)
    scale = np.ones_like(gross_arr, dtype=float)
    np.divide(max_gross, gross_arr, out=scale, where=gross_arr > max_gross)
    return pd.Series(pos.to_numpy(dtype=float) * scale, index=df.index)


def positions_model(df: pd.DataFrame, conf_th: float, q: float, max_gross: float = 1.0) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    score = df["pred_z"].to_numpy(dtype=float) * df["confidence"].to_numpy(dtype=float)
    tmp = df[["timestamp", "vol_24"]].copy()
    tmp["score"] = score
    tmp["idx"] = df.index
    for _, day in tmp.groupby("timestamp", sort=False):
        active = day[np.abs(day["score"]) > 1e-12]
        active = active[df.loc[active["idx"], "confidence"].to_numpy(dtype=float) >= conf_th]
        if len(active) < 4:
            continue
        long_cut = active["score"].quantile(1.0 - q)
        short_cut = active["score"].quantile(q)
        longs = active[active["score"] >= long_cut]
        shorts = active[active["score"] <= short_cut]
        for part, sign in [(longs, 1.0), (shorts, -1.0)]:
            if part.empty:
                continue
            weights = np.abs(part["score"].to_numpy(dtype=float)) / df.loc[part["idx"], "vol_24"].replace(0, np.nan).to_numpy(dtype=float)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            if weights.sum() <= 0:
                weights = np.ones(len(part))
            weights = weights / weights.sum() * (max_gross / 2.0)
            pos.loc[part["idx"]] = sign * weights
    return normalize_daily_gross(pos, df, max_gross=max_gross)


def positions_momentum(df: pd.DataFrame, q: float = 0.2) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    for _, day in df.groupby("timestamp", sort=False):
        if len(day) < 4:
            continue
        score = day["ret_12"] / day["vol_24"].replace(0, np.nan)
        long_cut = score.quantile(1.0 - q)
        short_cut = score.quantile(q)
        longs = day[score >= long_cut]
        shorts = day[score <= short_cut]
        if len(longs) > 0:
            pos.loc[longs.index] = 0.5 / len(longs)
        if len(shorts) > 0:
            pos.loc[shorts.index] = -0.5 / len(shorts)
    return pos


def positions_reversal(df: pd.DataFrame, q: float = 0.2) -> pd.Series:
    return -positions_momentum(df, q=q)


def positions_always_short(df: pd.DataFrame) -> pd.Series:
    return pd.Series(-1.0 / len(PAIRS), index=df.index)


def simulate(df: pd.DataFrame, position: pd.Series, name: str) -> dict[str, object]:
    work = df[["timestamp", "pair", "next_1bar_ret"]].copy()
    work["position"] = position.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    work["pnl"] = work["position"] * work["next_1bar_ret"]
    grouped = work.groupby("timestamp", sort=True)
    pnl = grouped["pnl"].sum()
    gross = grouped["position"].apply(lambda s: float(np.abs(s).sum()))
    short_exp = grouped["position"].apply(lambda s: float(np.abs(s[s < 0]).sum()))
    pos_wide = work.pivot(index="timestamp", columns="pair", values="position").fillna(0.0).sort_index()
    turnover = pos_wide.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = pos_wide.iloc[0].abs().sum()
    cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
    net = (pnl - cost).reindex(pnl.index).fillna(0.0)
    equity = INITIAL_CAPITAL * (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0) if len(equity) else 0.0
    vol = float(net.std(ddof=0))
    sharpe = float(net.mean() / vol * ANNUALIZATION) if vol > 0 else 0.0
    dd = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    trades = int((work["position"].abs() > 1e-12).sum())
    metrics: dict[str, object] = {
        "strategy": name,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(float(dd.min()) * 100 if len(dd) else 0.0, 2),
        "win_rate_pct": round(float((net > 0).mean()) * 100 if len(net) else 0.0, 1),
        "trades": trades,
        "final_value": round(float(equity.iloc[-1]) if len(equity) else INITIAL_CAPITAL, 2),
        "avg_gross_exposure": round(float(gross.mean()) if len(gross) else 0.0, 4),
        "avg_turnover": round(float(turnover.mean()) if len(turnover) else 0.0, 4),
        "avg_long_exposure": round(float(grouped["position"].apply(lambda s: float(s[s > 0].sum())).mean()) if len(work) else 0.0, 4),
        "avg_short_exposure": round(float(short_exp.mean()) if len(short_exp) else 0.0, 4),
        "score": round(sharpe + total_return + (float(dd.min()) if len(dd) else 0.0) - 0.03 * float(turnover.mean() if len(turnover) else 0.0), 4),
        "daily_returns": [float(x) for x in net.to_numpy()],
        "equity_curve": [float(x) for x in equity.to_numpy()],
    }
    return metrics


def select_model_params(select_df: pd.DataFrame) -> tuple[float, float, dict[str, object]]:
    best: tuple[float, float, dict[str, object]] | None = None
    for conf_th in [0.02, 0.08, 0.14, 0.20]:
        for q in [0.10, 0.20, 0.30]:
            pos = positions_model(select_df, conf_th=conf_th, q=q)
            metrics = simulate(select_df, pos, f"select(conf={conf_th},q={q})")
            if best is None or float(metrics["score"]) > float(best[2]["score"]):
                best = (conf_th, q, metrics)
    if best is None:
        raise RuntimeError("No model params selected")
    return best


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_results(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.6)
    plt.title("V12 4h Strategy Equity Curves")
    plt.ylabel("Equity")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v12_equity_curves.png", dpi=160)
    plt.close()

    names = list(results)
    returns = [float(results[n]["total_return_pct"]) for n in names]
    sharpes = [float(results[n]["sharpe"]) for n in names]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    ax1.bar(x - 0.18, returns, width=0.36, label="Return %")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, sharpes, width=0.36, color="orange", label="Sharpe")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("V12 4h Strategy Comparison")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v12_strategy_comparison.png", dpi=160)
    plt.close()


def write_report(results: dict[str, dict[str, object]], folds_log: list[dict[str, object]], config: dict[str, object]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results.values(), key=lambda m: float(m["score"]), reverse=True)
    best = ranked[0]
    lines = [
        "# V12 高频 4h 交易模型实验报告",
        "",
        "## 1. 实验目标",
        "V12 放弃日线 V4/V9/V10/V11 的预测缓存，重新使用 15 个主流币的 Binance 4h OHLCV 训练高频模型。目标是检验更短周期是否能改善日线模型在 regime 切换中的滞后问题。",
        "",
        "## 2. 方法",
        f"- 时间周期：{TIMEFRAME}",
        f"- 输入序列：{SEQ_LEN} 根 4h K 线，约 {SEQ_LEN / BAR_PER_DAY:.1f} 天",
        f"- 训练标签：未来 {LABEL_HORIZON} 根 4h K 线的成本调整方向与波动归一化收益",
        "- 模型：Transformer edge/confidence model，不直接输出自由 TP/SL 挂单",
        "- 执行：按 4h bar 重新调仓，使用 turnover-based fee+slippage，并按空头敞口计 funding",
        "- 验证：purged expanding walk-forward，训练/选择/交易窗口按时间严格分离",
        "",
        "## 3. 主要结果",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 交易数 | 平均总敞口 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in ranked:
        lines.append(
            f"| {m['strategy']} | {m['total_return_pct']}% | {m['sharpe']} | {m['max_drawdown_pct']}% | {m['trades']} | {m['avg_gross_exposure']} |"
        )
    lines += [
        "",
        "## 4. Fold 参数选择",
        "| Fold | fit 起止 | select 起止 | trade 起止 | conf | q | select score |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in folds_log:
        lines.append(
            f"| {row['fold']} | {row['fit_period']} | {row['select_period']} | {row['trade_period']} | {row['conf']} | {row['q']} | {row['select_score']} |"
        )
    lines += [
        "",
        "## 5. 结论",
        f"本次 V12 的候选冠军是 **{best['strategy']}**，收益 {best['total_return_pct']}%，Sharpe {best['sharpe']}，最大回撤 {best['max_drawdown_pct']}%。",
        "如果 V12-Model 未能稳定超过 Momentum/Reversal/AlwaysShort 等简单基线，则说明高频 Transformer 仍没有形成足够稳健的 alpha，应继续降低模型自由度或进入更严格的 paper trading 验证。",
        "",
        "## 6. 文件",
        f"- `{RESULTS_PATH.relative_to(ROOT)}`",
        f"- `{BACKTEST_PATH.relative_to(ROOT)}`",
        f"- `{DECISIONS_PATH.relative_to(ROOT)}`",
        "- `docs/figures/v12_equity_curves.png`",
        "- `docs/figures/v12_strategy_comparison.png`",
        "",
        "## 7. 配置",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def period(dates: list[pd.Timestamp]) -> str:
    return f"{dates[0]}~{dates[-1]}" if dates else "empty"


def main() -> None:
    set_seed()
    device = require_cuda()
    print("[1/7] Loading 4h cache", flush=True)
    raw = load_4h_cache()
    print("[2/7] Building 4h feature frame", flush=True)
    df = build_feature_frame(raw)
    dates = sorted(pd.to_datetime(df["timestamp"].unique()))
    folds = make_folds(dates)
    if not folds:
        raise RuntimeError("No V12 walk-forward folds built")
    print(f"features rows={len(df)} dates={len(dates)} folds={len(folds)} period={dates[0]} -> {dates[-1]}", flush=True)

    all_trade_parts: list[pd.DataFrame] = []
    folds_log: list[dict[str, object]] = []
    for fold in folds:
        print(f"[3/7] Fold {fold.fold}: fit={period(fold.fit_dates)} select={period(fold.select_dates)} trade={period(fold.trade_dates)}", flush=True)
        mean, std = fit_scaler(df, fold.fit_dates)
        x_fit, ya_fit, yr_fit, _ = build_sequences(df, fold.fit_dates, mean, std)
        x_select, _, _, row_select = build_sequences(df, fold.select_dates, mean, std)
        x_trade, _, _, row_trade = build_sequences(df, fold.trade_dates, mean, std)
        model = train_model(x_fit, ya_fit, yr_fit, device, fold.fold)
        select_pred = predict(model, x_select, row_select, device)
        trade_pred = predict(model, x_trade, row_trade, device)
        select_df = df.merge(select_pred, on="row_id", how="inner")
        trade_df = df.merge(trade_pred, on="row_id", how="inner")
        conf, q, selected = select_model_params(select_df)
        trade_df["pos_V12_Model"] = positions_model(trade_df, conf_th=conf, q=q).values
        trade_df["fold"] = fold.fold
        trade_df["selected_conf"] = conf
        trade_df["selected_q"] = q
        all_trade_parts.append(trade_df)
        folds_log.append({
            "fold": fold.fold,
            "fit_period": period(fold.fit_dates),
            "select_period": period(fold.select_dates),
            "trade_period": period(fold.trade_dates),
            "conf": conf,
            "q": q,
            "select_score": selected["score"],
        })
        print(f"fold {fold.fold} selected conf={conf:.2f} q={q:.2f} score={selected['score']}", flush=True)

    print("[4/7] Simulating strategies", flush=True)
    decisions = pd.concat(all_trade_parts, ignore_index=True).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    decisions["pos_Momentum"] = positions_momentum(decisions, q=0.2).values
    decisions["pos_Reversal"] = positions_reversal(decisions, q=0.2).values
    decisions["pos_AlwaysShort"] = positions_always_short(decisions).values
    decisions["pos_AlwaysFlat"] = 0.0

    results = {
        "V12-Model": simulate(decisions, decisions["pos_V12_Model"], "V12-Model"),
        "Momentum-4h": simulate(decisions, decisions["pos_Momentum"], "Momentum-4h"),
        "Reversal-4h": simulate(decisions, decisions["pos_Reversal"], "Reversal-4h"),
        "AlwaysShort-4h": simulate(decisions, decisions["pos_AlwaysShort"], "AlwaysShort-4h"),
        "AlwaysFlat": simulate(decisions, decisions["pos_AlwaysFlat"], "AlwaysFlat"),
    }
    ranked = sorted(results.values(), key=lambda m: float(m["score"]), reverse=True)
    for item in ranked:
        print(f"  {item['strategy']}: ret={item['total_return_pct']}% sharpe={item['sharpe']} mdd={item['max_drawdown_pct']}%", flush=True)

    print("[5/7] Saving files", flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(DECISIONS_PATH, index=False)
    config = {
        "timeframe": TIMEFRAME,
        "pairs": PAIRS,
        "seq_len": SEQ_LEN,
        "label_horizon_bars": LABEL_HORIZON,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "fee_rate": FEE_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "funding_daily": FUNDING_DAILY,
        "annualization": ANNUALIZATION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "rows": len(decisions),
        "dates": int(decisions["timestamp"].nunique()),
        "period": f"{decisions['timestamp'].min()} to {decisions['timestamp'].max()}",
    }
    RESULTS_PATH.write_text(json.dumps({"best_strategy": ranked[0]["strategy"], "results": [strip_series(x) for x in ranked], "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")
    BACKTEST_PATH.write_text(json.dumps({"results": {k: strip_series(v) for k, v in results.items()}, "folds": folds_log, "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[6/7] Plotting/reporting", flush=True)
    plot_results(results)
    write_report(results, folds_log, config)
    print(f"saved {RESULTS_PATH}", flush=True)
    print(f"saved {BACKTEST_PATH}", flush=True)
    print(f"saved {DECISIONS_PATH}", flush=True)
    print(f"saved {REPORT_PATH}", flush=True)
    print("[7/7] Done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
