#!/usr/bin/env python3
"""V10 Direct Trading Transformer.

This experiment makes the Transformer output executable trading decisions:
direction, position-size bucket, TP/SL bucket, horizon bucket and confidence.

The implementation intentionally keeps the trading layer constrained:
- no CPU fallback for neural training;
- no free continuous leverage output;
- no oracle TP/SL labels chosen from the best future path;
- walk-forward training with a fit/selection/trade split inside each fold.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/home/cyan/default/crypto")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_v9 as v9  # noqa: E402


INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_DAILY = 0.0001
SEQ_LEN = 45
FIT_EPOCHS = 70
BATCH_SIZE = 256
TRADE_WINDOW_DAYS = 30
MIN_TRAIN_DAYS = 90
SEED = 20260608
POS_SIZE = 1.0 / 15.0
SIZE_BUCKETS = np.array([0.0, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
TP_BUCKETS = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
SL_BUCKETS = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
HORIZON_BUCKETS = np.array([1, 3, 5], dtype=np.int64)

OUT_RESULTS = ROOT / "data/results_v10.json"
OUT_BACKTEST = ROOT / "data/backtest_v10_results.json"
OUT_TRADES = ROOT / "data/v10_trade_decisions.parquet"
OUT_REPORT = ROOT / "docs/v10_experiment_report.md"
FIG_DIR = ROOT / "docs/figures"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def require_cuda() -> torch.device:
    print("[CUDA] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[CUDA] torch=", torch.__version__, "available=", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V10; CPU fallback is forbidden.")
    for idx in range(torch.cuda.device_count()):
        print(f"[CUDA] cuda:{idx} {torch.cuda.get_device_name(idx)}")
    return torch.device("cuda:0")


def load_state_frame() -> tuple[pd.DataFrame, list[str]]:
    pred_path = ROOT / "data/v9_v4_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"]).dt.tz_localize(None)
    df, feature_cols = v9.add_state_features(pred)
    df = df.sort_values(["pair", "date"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df, feature_cols


def make_folds(dates: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]]:
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]] = []
    start = MIN_TRAIN_DAYS
    while start < len(dates):
        end = min(start + TRADE_WINDOW_DAYS, len(dates))
        train_dates = dates[:start]
        split = max(30, int(len(train_dates) * 0.72))
        fit_dates = train_dates[:split]
        select_dates = train_dates[split:]
        if len(select_dates) < 10:
            select_dates = train_dates[-30:]
            fit_dates = train_dates[:-30]
        folds.append((fit_dates, select_dates, dates[start:end]))
        start = end
    return folds


def scale_features(df: pd.DataFrame, feature_cols: list[str], fit_dates: list[pd.Timestamp]) -> pd.DataFrame:
    work = df.copy()
    fit_mask = work["date"].isin(set(fit_dates))
    means = work.loc[fit_mask, feature_cols].mean()
    stds = work.loc[fit_mask, feature_cols].std().replace(0, np.nan).fillna(1.0)
    work[feature_cols] = ((work[feature_cols] - means) / stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return work


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    threshold = FEE_RATE + SLIPPAGE_RATE + 0.0005
    ret = work["next_1d_ret"].astype(float)
    action = np.full(len(work), 1, dtype=np.int64)
    action[ret < -threshold] = 0
    action[ret > threshold] = 2
    work["label_action"] = action

    vol = work["vol_20"].replace(0, np.nan).fillna(work["vol_20"].median()).astype(float).clip(lower=0.003)
    risk_adj = (ret.abs() / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()
    work["label_size"] = np.digitize(risk_adj, bins=np.array([0.20, 0.45, 0.80, 1.20]), right=False).astype(np.int64)

    abs_pred = work["pred_3d"].abs().to_numpy()
    pred_ratio = abs_pred / np.maximum(vol.to_numpy(), 1e-4)
    work["label_tp"] = np.digitize(pred_ratio, bins=np.array([0.35, 0.70, 1.10]), right=False).astype(np.int64)
    work["label_sl"] = np.digitize(vol.to_numpy(), bins=np.nanquantile(vol.to_numpy(), [0.35, 0.65, 0.85]), right=False).astype(np.int64)

    abs_3d = work["actual_3d_ret"].abs().fillna(0.0).to_numpy()
    horizon = np.zeros(len(work), dtype=np.int64)
    horizon[abs_3d > np.maximum(abs_pred, 0.01)] = 1
    horizon[abs_3d > np.maximum(abs_pred * 1.5, 0.02)] = 2
    work["label_horizon"] = horizon
    work["target_ret"] = ret.astype(float)
    return work


def build_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    dates: list[pd.Timestamp],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], np.ndarray]:
    date_set = set(dates)
    sequences: list[np.ndarray] = []
    labels: dict[str, list[float]] = {
        "action": [],
        "size": [],
        "tp": [],
        "sl": [],
        "horizon": [],
        "ret": [],
    }
    row_ids: list[int] = []
    for _, pair_df in df.groupby("pair", sort=False):
        pair_df = pair_df.sort_values("date").reset_index(drop=True)
        values = pair_df[feature_cols].to_numpy(dtype=np.float32)
        for idx in range(SEQ_LEN - 1, len(pair_df)):
            row = pair_df.iloc[idx]
            if row["date"] not in date_set:
                continue
            sequences.append(values[idx - SEQ_LEN + 1 : idx + 1])
            labels["action"].append(float(row["label_action"]))
            labels["size"].append(float(row["label_size"]))
            labels["tp"].append(float(row["label_tp"]))
            labels["sl"].append(float(row["label_sl"]))
            labels["horizon"].append(float(row["label_horizon"]))
            labels["ret"].append(float(row["target_ret"]))
            row_ids.append(int(row["row_id"]))
    if not sequences:
        raise RuntimeError("No samples built for requested dates")
    x = torch.tensor(np.stack(sequences), dtype=torch.float32)
    y = {
        "action": torch.tensor(labels["action"], dtype=torch.long),
        "size": torch.tensor(labels["size"], dtype=torch.long),
        "tp": torch.tensor(labels["tp"], dtype=torch.long),
        "sl": torch.tensor(labels["sl"], dtype=torch.long),
        "horizon": torch.tensor(labels["horizon"], dtype=torch.long),
        "ret": torch.tensor(labels["ret"], dtype=torch.float32),
    }
    return x, y, np.asarray(row_ids, dtype=np.int64)


class DirectTradingTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 96, n_heads: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.action = nn.Linear(d_model, 3)
        self.size = nn.Linear(d_model, len(SIZE_BUCKETS))
        self.tp = nn.Linear(d_model, len(TP_BUCKETS))
        self.sl = nn.Linear(d_model, len(SL_BUCKETS))
        self.horizon = nn.Linear(d_model, len(HORIZON_BUCKETS))
        self.ret = nn.Linear(d_model, 1)
        self.conf = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        z = self.encoder(z)
        pooled = self.norm(z[:, -1])
        return {
            "action": self.action(pooled),
            "size": self.size(pooled),
            "tp": self.tp(pooled),
            "sl": self.sl(pooled),
            "horizon": self.horizon(pooled),
            "ret": self.ret(pooled).squeeze(-1),
            "conf": self.conf(pooled).squeeze(-1),
        }


def train_fold_model(
    model: nn.Module,
    x: torch.Tensor,
    y: dict[str, torch.Tensor],
    device: torch.device,
    fold_id: int,
) -> nn.Module:
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    ds = TensorDataset(x, y["action"], y["size"], y["tp"], y["sl"], y["horizon"], y["ret"])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-3)
    for epoch in range(1, FIT_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            xb, ya, ys, ytp, ysl, yh, yr = [item.to(device) for item in batch]
            out = model(xb)
            loss = (
                F.cross_entropy(out["action"], ya)
                + 0.35 * F.cross_entropy(out["size"], ys)
                + 0.12 * F.cross_entropy(out["tp"], ytp)
                + 0.12 * F.cross_entropy(out["sl"], ysl)
                + 0.08 * F.cross_entropy(out["horizon"], yh)
                + 7.5 * F.huber_loss(out["ret"], yr, delta=0.02)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {1, 10, 25, 50, FIT_EPOCHS}:
            print(f"  fold={fold_id} epoch={epoch:03d} loss={np.mean(losses):.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, row_ids: np.ndarray, device: torch.device) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=512, shuffle=False, num_workers=0)
    rows: list[pd.DataFrame] = []
    start = 0
    for (xb,) in loader:
        xb = xb.to(device)
        out = model(xb)
        action_prob = torch.softmax(out["action"], dim=1).cpu().numpy()
        size_prob = torch.softmax(out["size"], dim=1).cpu().numpy()
        tp_prob = torch.softmax(out["tp"], dim=1).cpu().numpy()
        sl_prob = torch.softmax(out["sl"], dim=1).cpu().numpy()
        horizon_prob = torch.softmax(out["horizon"], dim=1).cpu().numpy()
        n = len(action_prob)
        part = pd.DataFrame({
            "row_id": row_ids[start : start + n],
            "p_short": action_prob[:, 0],
            "p_flat": action_prob[:, 1],
            "p_long": action_prob[:, 2],
            "size_expected": size_prob @ SIZE_BUCKETS,
            "tp_expected": tp_prob @ TP_BUCKETS,
            "sl_expected": sl_prob @ SL_BUCKETS,
            "horizon_expected": horizon_prob @ HORIZON_BUCKETS,
            "pred_ret_v10": out["ret"].cpu().numpy(),
            "raw_conf": torch.sigmoid(out["conf"]).cpu().numpy(),
        })
        rows.append(part)
        start += n
    pred = pd.concat(rows, ignore_index=True)
    edge = pred["p_long"] - pred["p_short"]
    pred["edge"] = edge
    pred["confidence"] = (pred[["p_short", "p_long"]].max(axis=1) * (1.0 - pred["p_flat"]) * pred["raw_conf"]).clip(0, 1)
    return pred


def normalize_daily(weights: pd.Series, df: pd.DataFrame, max_gross: float = 1.0) -> pd.Series:
    out = weights.copy().astype(float)
    for date, idx in df.groupby("date").groups.items():
        gross = float(np.abs(out.loc[idx]).sum())
        if gross > max_gross:
            out.loc[idx] = out.loc[idx] / gross * max_gross
    return out


def positions_from_predictions(df: pd.DataFrame, mode: str, threshold: float = 0.05, q: float = 0.2) -> pd.Series:
    pos = pd.Series(0.0, index=df.index)
    if mode == "A":
        pos[df["edge"] > threshold] = POS_SIZE
        pos[df["edge"] < -threshold] = -POS_SIZE
        return normalize_daily(pos, df)
    if mode == "B":
        raw = np.sign(df["edge"].to_numpy()) * df["size_expected"].to_numpy() * POS_SIZE
        raw[np.abs(df["edge"].to_numpy()) <= threshold] = 0.0
        vol = df["vol_20"].replace(0, np.nan).fillna(df["vol_20"].median()).clip(lower=0.004).to_numpy()
        raw = raw * np.clip(0.03 / vol, 0.25, 2.0)
        return normalize_daily(pd.Series(raw, index=df.index), df)
    if mode == "C":
        raw = np.sign(df["edge"].to_numpy()) * df["size_expected"].to_numpy() * df["confidence"].to_numpy() * POS_SIZE
        raw[np.abs(df["edge"].to_numpy()) <= threshold] = 0.0
        return normalize_daily(pd.Series(raw, index=df.index), df)
    if mode == "D":
        score = df["edge"].to_numpy() * (0.5 + df["size_expected"].to_numpy()) * df["confidence"].to_numpy()
        vol = df["vol_20"].replace(0, np.nan).fillna(df["vol_20"].median()).clip(lower=0.004).to_numpy()
        score = score / vol
        for _, day in df.assign(score=score).groupby("date"):
            n = len(day)
            k = max(1, int(round(n * q)))
            ordered = day.sort_values("score", ascending=False)
            long_idx = ordered.head(k).index
            short_idx = ordered.tail(k).index
            long_score = np.maximum(ordered.head(k)["score"].to_numpy(), 0.0)
            short_score = np.maximum(-ordered.tail(k)["score"].to_numpy(), 0.0)
            if long_score.sum() <= 0:
                long_score = np.ones(k)
            if short_score.sum() <= 0:
                short_score = np.ones(k)
            pos.loc[long_idx] = 0.5 * long_score / long_score.sum()
            pos.loc[short_idx] = -0.5 * short_score / short_score.sum()
        return normalize_daily(pos, df)
    raise ValueError(mode)


def simulate(
    df: pd.DataFrame,
    positions: pd.Series,
    name: str,
    clipped: bool = False,
) -> dict[str, object]:
    work = df[["date", "pair", "next_1d_ret", "vol_20", "tp_expected", "sl_expected"]].copy()
    work["position"] = positions.reindex(work.index).fillna(0.0).to_numpy(dtype=float)
    if clipped:
        vol = work["vol_20"].replace(0, np.nan).fillna(work["vol_20"].median()).clip(lower=0.004).to_numpy()
        tp = work["tp_expected"].to_numpy() * vol
        sl = work["sl_expected"].to_numpy() * vol
        signed_ret = work["next_1d_ret"].to_numpy(dtype=float)
        pos = work["position"].to_numpy(dtype=float)
        effective = signed_ret.copy()
        long_mask = pos > 0
        short_mask = pos < 0
        effective[long_mask] = np.clip(effective[long_mask], -sl[long_mask], tp[long_mask])
        effective[short_mask] = np.clip(effective[short_mask], -tp[short_mask], sl[short_mask])
        work["trade_ret"] = effective
    else:
        work["trade_ret"] = work["next_1d_ret"].astype(float)

    dates = sorted(work["date"].unique())
    pv = [INITIAL_CAPITAL]
    daily_returns: list[float] = []
    gross_values: list[float] = []
    long_values: list[float] = []
    short_values: list[float] = []
    trades = 0
    for date in dates:
        day = work[work["date"] == date]
        weights = day["position"].to_numpy(dtype=float)
        gross = float(np.abs(weights).sum())
        long_exp = float(np.clip(weights, 0, None).sum())
        short_exp = float(np.clip(-weights, 0, None).sum())
        pnl = float(np.sum(weights * day["trade_ret"].to_numpy(dtype=float)))
        cost = (FEE_RATE + SLIPPAGE_RATE) * gross + FUNDING_DAILY * short_exp
        net = pnl - cost
        pv.append(pv[-1] * (1.0 + net))
        daily_returns.append(net)
        gross_values.append(gross)
        long_values.append(long_exp)
        short_values.append(short_exp)
        trades += int(np.count_nonzero(np.abs(weights) > 1e-12))
    pv_arr = np.asarray(pv, dtype=float)
    dr = np.asarray(daily_returns, dtype=float)
    sharpe = float(dr.mean() / (dr.std() + 1e-10) * math.sqrt(365)) if len(dr) else 0.0
    win = float((dr > 0).mean() * 100.0) if len(dr) else 0.0
    peak = np.maximum.accumulate(pv_arr)
    mdd = float(((pv_arr - peak) / peak).min() * 100.0)
    total = float((pv_arr[-1] / pv_arr[0] - 1.0) * 100.0)
    score = sharpe + total / 100.0 + mdd / 100.0 - min(trades / 100_000.0, 0.1)
    return {
        "strategy": name,
        "total_return_pct": round(total, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(win, 1),
        "trades": int(trades),
        "final_value": round(float(pv_arr[-1]), 2),
        "avg_gross_exposure": round(float(np.mean(gross_values)) if gross_values else 0.0, 4),
        "avg_long_exposure": round(float(np.mean(long_values)) if long_values else 0.0, 4),
        "avg_short_exposure": round(float(np.mean(short_values)) if short_values else 0.0, 4),
        "score": round(score, 4),
        "daily_returns": daily_returns,
        "equity_curve": pv,
    }


def select_params(df: pd.DataFrame, mode: str) -> tuple[float, float, dict[str, object]]:
    best: tuple[float, float, dict[str, object]] | None = None
    thresholds = [0.00, 0.03, 0.06, 0.10, 0.15, 0.20]
    qs = [0.10, 0.20, 0.30]
    for th in thresholds:
        q_values = qs if mode == "D" else [0.2]
        for q in q_values:
            pos = positions_from_predictions(df, mode=mode, threshold=th, q=q)
            metrics = simulate(df, pos, f"select-{mode}", clipped=(mode == "C"))
            candidate = (th, q, metrics)
            if best is None or float(metrics["score"]) > float(best[2]["score"]):
                best = candidate
    if best is None:
        raise RuntimeError("No params selected")
    return best


def baseline_positions(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "V4-LS(th=0.005)": v9.positions_ls(df, threshold=0.005),
        "V9-VolTarget-Quantile(q=0.20)": v9.positions_vol_target(df, threshold=0.0, q=0.20),
        "AlwaysShort(daily_fee)": v9.positions_always(df, "short"),
        "RandomLS(seed=7)": v9.positions_random(df, seed=7),
    }


def positions_gated_v9(df: pd.DataFrame, conf_th: float, require_agreement: bool) -> pd.Series:
    base = v9.positions_vol_target(df, threshold=0.0, q=0.20).astype(float)
    conf = df["confidence"].to_numpy(dtype=float)
    edge = df["edge"].to_numpy(dtype=float)
    if require_agreement:
        keep = (np.sign(base.to_numpy(dtype=float)) == np.sign(edge)) & (conf >= conf_th)
    else:
        keep = conf >= conf_th
    out = base.copy()
    out.loc[~keep] = 0.0
    return normalize_daily(out, df)


def select_gated_v9_params(df: pd.DataFrame) -> tuple[float, bool, dict[str, object]]:
    best: tuple[float, bool, dict[str, object]] | None = None
    for conf_th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for require_agreement in [False, True]:
            pos = positions_gated_v9(df, conf_th=conf_th, require_agreement=require_agreement)
            metrics = simulate(df, pos, "select-gated-v9")
            candidate = (conf_th, require_agreement, metrics)
            if best is None or float(metrics["score"]) > float(best[2]["score"]):
                best = candidate
    if best is None:
        raise RuntimeError("No V10-F params selected")
    return best


def clean_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_results(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    top = sorted(results.values(), key=lambda m: float(m["score"]), reverse=True)[:10]
    plt.figure(figsize=(12, 7))
    for metrics in top[:8]:
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=str(metrics["strategy"]))
    plt.title("V10 Direct Trading Transformer Equity Curves")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v10_equity_curves.png", dpi=160)
    plt.close()

    names = [str(m["strategy"]) for m in top]
    returns = [float(m["total_return_pct"]) for m in top]
    sharpes = [float(m["sharpe"]) for m in top]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(names[::-1], returns[::-1])
    axes[0].set_title("Total Return %")
    axes[1].barh(names[::-1], sharpes[::-1])
    axes[1].set_title("Sharpe")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v10_strategy_comparison.png", dpi=160)
    plt.close()


def write_report(results: dict[str, dict[str, object]], fold_log: list[dict[str, object]]) -> None:
    top = sorted([clean_metrics(m) for m in results.values()], key=lambda m: float(m["score"]), reverse=True)
    best = top[0]
    lines = [
        "# V10：Direct Trading Transformer 实验报告",
        "",
        "## 1. 实验目标",
        "",
        "V10 让 Transformer 直接输出交易动作：做多/做空/空仓、仓位档位、止盈止损档位、持仓周期和置信度。",
        "本实验不是单纯预测收益率，而是按交易结果评估模型。",
        "",
        "## 2. 训练与执行约束",
        "",
        f"- CUDA_VISIBLE_DEVICES：`{os.environ.get('CUDA_VISIBLE_DEVICES')}`。",
        f"- PyTorch：`{torch.__version__}`。",
        "- 信号输入来自 V9 保存的 V4 明细预测 `data/v9_v4_predictions.parquet`。",
        "- 使用 expanding walk-forward：每个 fold 用历史 fit window 训练，用 selection window 选阈值/分位数，再在未来 30 天交易。",
        "- 成本包括 fee、slippage 和简化 funding/short cost。",
        "- TP/SL 不使用未来最优路径标签，只输出固定波动率网格档位；同日路径不可知问题按保守近似处理。",
        "",
        "## 3. 主要结果",
        "",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in top[:16]:
        lines.append(
            f"| {m['strategy']} | {m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | "
            f"{m['max_drawdown_pct']:.2f}% | {m['win_rate_pct']:.1f}% | {m['trades']} | {m['score']:+.3f} |"
        )
    lines.extend([
        "",
        "## 4. 最优候选判断",
        "",
        f"本轮 V10 排名第一的是 **{best['strategy']}**：收益 {best['total_return_pct']:+.2f}%，Sharpe {best['sharpe']:.2f}，最大回撤 {best['max_drawdown_pct']:.2f}%。",
        "",
        "需要强调：V10 的目标是检验 Transformer 直接交易是否能超过 V9。若最佳结果仍低于 V9-VolTarget-Quantile(q=0.20)，则说明直接交易 Transformer 暂时没有证明优于简单交易层。",
        "",
        "## 5. Fold 参数选择",
        "",
        "| Fold | 交易区间 | A阈值 | B阈值 | C阈值 | D阈值 | D分位数 | F置信度 | F需同向 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for item in fold_log:
        lines.append(
            f"| {item['fold']} | {item['trade_start']}~{item['trade_end']} | "
            f"{item['A_th']:.2f} | {item['B_th']:.2f} | {item['C_th']:.2f} | {item['D_th']:.2f} | "
            f"{item['D_q']:.2f} | {item['F_conf']:.2f} | {item['F_agree']} |"
        )
    lines.extend([
        "",
        "## 6. 结论",
        "",
        "V10 已经实现直接交易输出，但是否值得继续取决于它是否稳定超过 V9 简单规则候选。",
        "如果 V10 低于 V9，下一轮改进应优先做：更严格的成本模型、增加组合状态、加入收益分布头的校准损失、以及用 V9 规则做 teacher distillation，而不是直接加大模型。",
    ])
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    device = require_cuda()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading V9/V4 prediction state frame...", flush=True)
    base_df, feature_cols = load_state_frame()
    base_df = label_frame(base_df)
    all_dates = sorted(pd.to_datetime(base_df["date"].unique()))
    folds = make_folds(all_dates)
    print(f"  rows={len(base_df)} dates={len(all_dates)} features={len(feature_cols)} folds={len(folds)}", flush=True)

    all_trade_parts: list[pd.DataFrame] = []
    fold_log: list[dict[str, object]] = []

    for fold_id, (fit_dates, select_dates, trade_dates) in enumerate(folds, start=1):
        print(
            f"\n[2/6] Fold {fold_id}/{len(folds)} fit={fit_dates[0].date()}~{fit_dates[-1].date()} "
            f"select={select_dates[0].date()}~{select_dates[-1].date()} trade={trade_dates[0].date()}~{trade_dates[-1].date()}",
            flush=True,
        )
        scaled = scale_features(base_df, feature_cols, fit_dates)
        x_fit, y_fit, _ = build_samples(scaled, feature_cols, fit_dates)
        x_select, _, select_ids = build_samples(scaled, feature_cols, select_dates)
        x_trade, _, trade_ids = build_samples(scaled, feature_cols, trade_dates)
        model = DirectTradingTransformer(n_features=len(feature_cols))
        model = train_fold_model(model, x_fit, y_fit, device, fold_id)
        select_pred = predict(model, x_select, select_ids, device)
        trade_pred = predict(model, x_trade, trade_ids, device)
        select_df = base_df.merge(select_pred, on="row_id", how="inner")
        trade_df = base_df.merge(trade_pred, on="row_id", how="inner")
        params = {mode: select_params(select_df, mode) for mode in ["A", "B", "C", "D"]}
        for mode, (th, q, metric) in params.items():
            print(f"  select V10-{mode}: th={th:.2f} q={q:.2f} score={metric['score']}", flush=True)
            pos = positions_from_predictions(trade_df, mode=mode, threshold=th, q=q)
            trade_df[f"pos_V10_{mode}"] = pos.reindex(trade_df.index).fillna(0.0).values
        f_conf, f_agree, f_metric = select_gated_v9_params(select_df)
        print(f"  select V10-F: conf={f_conf:.2f} agree={f_agree} score={f_metric['score']}", flush=True)
        f_pos = positions_gated_v9(trade_df, conf_th=f_conf, require_agreement=f_agree)
        trade_df["pos_V10_F"] = f_pos.reindex(trade_df.index).fillna(0.0).values
        fold_log.append({
            "fold": fold_id,
            "trade_start": str(trade_dates[0].date()),
            "trade_end": str(trade_dates[-1].date()),
            "A_th": params["A"][0],
            "B_th": params["B"][0],
            "C_th": params["C"][0],
            "D_th": params["D"][0],
            "D_q": params["D"][1],
            "F_conf": f_conf,
            "F_agree": f_agree,
        })
        all_trade_parts.append(trade_df)

    print("\n[3/6] Aggregating walk-forward trade decisions...", flush=True)
    decisions = pd.concat(all_trade_parts, ignore_index=True).sort_values(["date", "pair"]).reset_index(drop=True)
    decisions.to_parquet(OUT_TRADES, index=False)
    eval_dates = sorted(pd.to_datetime(decisions["date"].unique()))
    print(f"  decisions={len(decisions)} dates={len(eval_dates)} file={OUT_TRADES}", flush=True)

    print("[4/6] Evaluating V10 and baselines...", flush=True)
    results: dict[str, dict[str, object]] = {}
    results["V10-A ActionOnly"] = simulate(decisions, decisions["pos_V10_A"], "V10-A ActionOnly")
    results["V10-B ActionSize"] = simulate(decisions, decisions["pos_V10_B"], "V10-B ActionSize")
    results["V10-C ActionSizeTPSL"] = simulate(decisions, decisions["pos_V10_C"], "V10-C ActionSizeTPSL", clipped=True)
    results["V10-D DirectQuantile"] = simulate(decisions, decisions["pos_V10_D"], "V10-D DirectQuantile")
    results["V10-F TransformerGatedV9"] = simulate(decisions, decisions["pos_V10_F"], "V10-F TransformerGatedV9")
    for name, pos in baseline_positions(decisions).items():
        results[name] = simulate(decisions, pos, name)

    sorted_results = sorted(results.values(), key=lambda m: float(m["score"]), reverse=True)
    for metrics in sorted_results:
        print(
            f"  {metrics['strategy']}: ret={metrics['total_return_pct']:+.2f}% "
            f"sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown_pct']:.2f}% score={metrics['score']:+.3f}",
            flush=True,
        )

    print("[5/6] Writing outputs...", flush=True)
    clean = {name: clean_metrics(m) for name, m in results.items()}
    best = max(clean.values(), key=lambda m: float(m["score"]))
    OUT_BACKTEST.write_text(json.dumps({
        "config": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "seq_len": SEQ_LEN,
            "fit_epochs": FIT_EPOCHS,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "funding_daily": FUNDING_DAILY,
        },
        "fold_log": fold_log,
        "strategies": clean,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_RESULTS.write_text(json.dumps({
        "best_strategy": best,
        "top": sorted(clean.values(), key=lambda m: float(m["score"]), reverse=True),
        "config": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "trade_period": f"{eval_dates[0].date()} to {eval_dates[-1].date()}",
            "rows": len(decisions),
            "dates": len(eval_dates),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(results)
    write_report(results, fold_log)
    print(f"  saved {OUT_RESULTS}")
    print(f"  saved {OUT_BACKTEST}")
    print(f"  saved {OUT_REPORT}")
    print("[6/6] Done.", flush=True)


if __name__ == "__main__":
    main()
