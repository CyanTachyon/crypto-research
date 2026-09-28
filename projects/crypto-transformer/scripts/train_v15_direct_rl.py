#!/usr/bin/env python3
"""V15: direct RL-style actor-only portfolio trader.

This experiment intentionally does NOT use V12/V13/V14 prediction columns as
state.  It trains a PyTorch policy directly from 4h OHLCV-derived features to
target portfolio weights across 15 pairs.

Strict split:
- train/fit: <= 2024-07-01
- select:    2024-07-23 .. 2024-12-10
- holdout:   >= 2025-01-01, evaluated once with frozen model/params
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

INPUT = DATA_DIR / "v12_trade_decisions.parquet"
DECISIONS_OUT = DATA_DIR / "v15_direct_rl_decisions.parquet"
RESULTS_OUT = DATA_DIR / "results_v15_direct_rl.json"
BACKTEST_OUT = DATA_DIR / "backtest_v15_direct_rl_results.json"
REPORT_OUT = DOCS_DIR / "v15_direct_rl_report.md"
CHECKPOINT_OUT = DATA_DIR / "checkpoints" / "v15_direct_rl_policy.pt"

INITIAL_CAPITAL = 10_000.0
BAR_PER_DAY = 6
ANNUALIZATION = math.sqrt(365 * BAR_PER_DAY)
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_PER_BAR = 0.0001 / BAR_PER_DAY

FIT_END = pd.Timestamp("2024-07-01 00:00:00")
SELECT_START = pd.Timestamp("2024-07-23 00:00:00")
SELECT_END = pd.Timestamp("2024-12-10 00:00:00")
HOLDOUT_START = pd.Timestamp("2025-01-01 00:00:00")

SEED = 20260608
EPOCHS = 35
TRAIN_MAX_STEPS = 1_200
LR = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN = 128
DROPOUT = 0.12
PAIR_CAP = 0.04
MAX_GROSS_GRID = [0.10, 0.15]
TEMP_GRID = [1.00]
TURNOVER_PENALTY_GRID = [2.0]
DD_PENALTY = 1.5
VOL_PENALTY = 0.25
ENTROPY_PENALTY = 0.002

MARKET_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "btc_ret_6", "btc_ret_24", "btc_vol_24", "market_ret_6", "market_vol_24",
]

PAIR_FEATURES = [
    "pair_id",
    "log_ret_1", "hl_range", "oc_ret", "volume_log", "volume_ratio_24",
    "ret_3", "ret_6", "ret_12", "ret_24", "ret_42", "ret_72",
    "vol_6", "vol_12", "vol_24", "vol_42", "vol_72",
    "range_6", "range_24", "price_pos_24", "price_pos_72",
    "mom_z_12", "mom_z_24", "vol_ratio_6_24",
    "ret_rank_6", "ret_z_cs", "vol_rank_24",
]

FORBIDDEN_STATE_COLS = {
    "p_short", "p_flat", "p_long", "pred_z", "raw_conf", "edge", "confidence", "score",
    "xgb_pred_z", "xgb_p_short", "xgb_p_flat", "xgb_p_long", "xgb_edge", "xgb_conf",
    "pos_V12_Model", "pos_V12_LowTurnover", "pos_V12_LowTurnover_Exploratory",
    "pos_V13_Target", "pos_V13_EnsembleRouterRisk", "pos_V13_PostHocSelector", "pos_V13_ConservativeSelector",
}


@dataclass(frozen=True)
class DataBundle:
    pairs: list[str]
    fit_idx: np.ndarray
    select_idx: np.ndarray
    holdout_idx: np.ndarray
    pair_features: np.ndarray
    market_features: np.ndarray
    returns: np.ndarray
    vol_24: np.ndarray
    timestamps: list[pd.Timestamp]
    raw_df: pd.DataFrame


class DirectRLPolicy(nn.Module):
    def __init__(self, pair_dim: int, market_dim: int, hidden: int) -> None:
        super().__init__()
        self.pair_net = nn.Sequential(
            nn.Linear(pair_dim + market_dim + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden, 1)

    def forward(self, pair_x: torch.Tensor, market_x: torch.Tensor, prev_pos: torch.Tensor, drawdown: torch.Tensor) -> torch.Tensor:
        # pair_x: [T, P, F], market_x: [T, M], prev_pos: [T, P], drawdown: [T]
        t, p, _ = pair_x.shape
        market = market_x[:, None, :].expand(t, p, market_x.shape[-1])
        account = torch.stack([prev_pos, prev_pos.abs(), drawdown[:, None].expand(t, p)], dim=-1)
        x = torch.cat([pair_x, market, account], dim=-1)
        h = self.pair_net(x)
        return self.score_head(h).squeeze(-1)


def require_cuda() -> torch.device:
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V15; CPU fallback forbidden")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)
    return torch.device("cuda")


def load_input() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run V12 first")
    df = pd.read_parquet(INPUT).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    required = {"timestamp", "pair", "next_1bar_ret", "vol_24", *PAIR_FEATURES, *MARKET_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    used = set(PAIR_FEATURES + MARKET_FEATURES)
    leakage = sorted(used & FORBIDDEN_STATE_COLS)
    if leakage:
        raise RuntimeError(f"Forbidden signal columns in state: {leakage}")
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).reset_index(drop=True)


def make_bundle(df: pd.DataFrame) -> DataBundle:
    pairs = sorted(df["pair"].unique())
    timestamps = sorted(df["timestamp"].unique())
    pair_to_i = {p: i for i, p in enumerate(pairs)}
    time_to_i = {ts: i for i, ts in enumerate(timestamps)}
    t_len, p_len = len(timestamps), len(pairs)
    pair_features = np.zeros((t_len, p_len, len(PAIR_FEATURES)), dtype=np.float32)
    returns = np.zeros((t_len, p_len), dtype=np.float32)
    vol_24 = np.zeros((t_len, p_len), dtype=np.float32)
    for row in df.itertuples(index=False):
        ti = time_to_i[row.timestamp]
        pi = pair_to_i[row.pair]
        pair_features[ti, pi, :] = [float(getattr(row, c)) for c in PAIR_FEATURES]
        returns[ti, pi] = float(row.next_1bar_ret)
        vol_24[ti, pi] = float(row.vol_24)
    market_df = df.groupby("timestamp")[MARKET_FEATURES].first().reindex(timestamps)
    market_features = market_df.to_numpy(dtype=np.float32)

    ts_arr = np.array(timestamps, dtype="datetime64[ns]")
    fit_idx = np.flatnonzero(ts_arr <= np.datetime64(FIT_END))
    select_idx = np.flatnonzero((ts_arr >= np.datetime64(SELECT_START)) & (ts_arr <= np.datetime64(SELECT_END)))
    holdout_idx = np.flatnonzero(ts_arr >= np.datetime64(HOLDOUT_START))
    if len(fit_idx) == 0 or len(select_idx) == 0 or len(holdout_idx) == 0:
        raise RuntimeError("Empty V15 split")
    return DataBundle(pairs, fit_idx, select_idx, holdout_idx, pair_features, market_features, returns, vol_24, timestamps, df)


def standardize_bundle(bundle: DataBundle) -> DataBundle:
    pf = bundle.pair_features.copy()
    mf = bundle.market_features.copy()
    fit_pf = pf[bundle.fit_idx].reshape(-1, pf.shape[-1])
    fit_mf = mf[bundle.fit_idx]
    pf_mean = fit_pf.mean(axis=0)
    pf_std = fit_pf.std(axis=0) + 1e-6
    mf_mean = fit_mf.mean(axis=0)
    mf_std = fit_mf.std(axis=0) + 1e-6
    pf = np.clip((pf - pf_mean) / pf_std, -8.0, 8.0).astype(np.float32)
    mf = np.clip((mf - mf_mean) / mf_std, -8.0, 8.0).astype(np.float32)
    return DataBundle(bundle.pairs, bundle.fit_idx, bundle.select_idx, bundle.holdout_idx, pf, mf, bundle.returns, bundle.vol_24, bundle.timestamps, bundle.raw_df)


def project_weights(scores: torch.Tensor, max_gross: float, temperature: float) -> torch.Tensor:
    centered = scores - scores.mean(dim=1, keepdim=True)
    raw = torch.tanh(centered / max(temperature, 1e-3))
    raw = torch.clamp(raw, -PAIR_CAP, PAIR_CAP)
    gross = raw.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(max_gross / gross, max=1.0)
    return raw * scale


def differentiable_run(
    model: DirectRLPolicy,
    pair_x: torch.Tensor,
    market_x: torch.Tensor,
    returns: torch.Tensor,
    *,
    max_gross: float,
    temperature: float,
    turnover_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t_len, p_len = returns.shape
    prev = torch.zeros(p_len, device=returns.device)
    equity = torch.ones((), device=returns.device)
    peak = torch.ones((), device=returns.device)
    nets = []
    objective_terms = []
    turnovers = []
    gross_list = []
    positions = []
    drawdown = torch.zeros((), device=returns.device)
    for i in range(t_len):
        score = model(pair_x[i:i+1], market_x[i:i+1], prev[None, :], drawdown[None])[0]
        pos = project_weights(score[None, :], max_gross, temperature)[0]
        turnover = (pos - prev).abs().sum()
        short_exp = pos.clamp(max=0.0).abs().sum()
        pnl = (pos * returns[i]).sum()
        cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
        real_net = pnl - cost
        risk = VOL_PENALTY * pnl.pow(2) + DD_PENALTY * F.relu(-0.08 - drawdown).pow(2)
        turnover_obj = turnover_penalty * turnover * 0.0005
        entropy_obj = ENTROPY_PENALTY * (pos.abs().sum() / max(max_gross, 1e-6))
        objective_net = real_net - turnover_obj - risk - entropy_obj
        equity = equity * (1.0 + real_net).clamp_min(0.80)
        peak = torch.maximum(peak, equity)
        drawdown = equity / peak - 1.0
        nets.append(real_net)
        objective_terms.append(objective_net)
        turnovers.append(turnover)
        gross_list.append(pos.abs().sum())
        positions.append(pos)
        prev = pos
    net_t = torch.stack(nets)
    obj_t = torch.stack(objective_terms)
    mean = obj_t.mean()
    std = net_t.std(unbiased=False).clamp_min(1e-6)
    sharpe_like = mean / std * ANNUALIZATION
    total = torch.prod((1.0 + net_t).clamp_min(0.80)) - 1.0
    loss = -(sharpe_like + 0.4 * total - 0.05 * torch.stack(turnovers).mean() - 0.02 * torch.stack(gross_list).mean())
    return loss, {
        "net": net_t,
        "positions": torch.stack(positions),
        "turnover": torch.stack(turnovers),
        "gross": torch.stack(gross_list),
    }


def tensors_for_idx(bundle: DataBundle, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(bundle.pair_features[idx], dtype=torch.float32, device=device),
        torch.tensor(bundle.market_features[idx], dtype=torch.float32, device=device),
        torch.tensor(bundle.returns[idx], dtype=torch.float32, device=device),
    )


def train_one(bundle: DataBundle, device: torch.device, max_gross: float, temperature: float, turnover_penalty: float, seed: int) -> DirectRLPolicy:
    torch.manual_seed(seed)
    model = DirectRLPolicy(len(PAIR_FEATURES), len(MARKET_FEATURES), HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_idx = bundle.fit_idx[-TRAIN_MAX_STEPS:]
    pair_x, market_x, returns = tensors_for_idx(bundle, train_idx, device)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, stats = differentiable_run(model, pair_x, market_x, returns, max_gross=max_gross, temperature=temperature, turnover_penalty=turnover_penalty)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch in {1, 10, 20, 35}:
            net = stats["net"].detach()
            print(
                f"    epoch {epoch:03d} loss={float(loss.detach()):+.5f} mean_net={float(net.mean()):+.6f} "
                f"gross={float(stats['gross'].mean().detach()):.4f} turn={float(stats['turnover'].mean().detach()):.4f}",
                flush=True,
            )
    return model


@torch.no_grad()
def infer_positions(bundle: DataBundle, idx: np.ndarray, model: DirectRLPolicy, device: torch.device, max_gross: float, temperature: float) -> np.ndarray:
    model.eval()
    pair_x, market_x, _ = tensors_for_idx(bundle, idx, device)
    prev = torch.zeros(len(bundle.pairs), device=device)
    drawdown = torch.zeros((), device=device)
    equity = torch.ones((), device=device)
    peak = torch.ones((), device=device)
    positions = []
    returns_t = torch.tensor(bundle.returns[idx], dtype=torch.float32, device=device)
    for i in range(len(idx)):
        score = model(pair_x[i:i+1], market_x[i:i+1], prev[None, :], drawdown[None])[0]
        pos = project_weights(score[None, :], max_gross, temperature)[0]
        pnl = (pos * returns_t[i]).sum()
        turnover = (pos - prev).abs().sum()
        short_exp = pos.clamp(max=0.0).abs().sum()
        cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
        equity = equity * (1.0 + pnl - cost).clamp_min(0.80)
        peak = torch.maximum(peak, equity)
        drawdown = equity / peak - 1.0
        positions.append(pos.detach().cpu().numpy())
        prev = pos
    return np.asarray(positions, dtype=np.float32)


def simulate_matrix(returns: np.ndarray, positions: np.ndarray, name: str) -> dict[str, object]:
    prev = np.zeros(positions.shape[1], dtype=np.float64)
    net_values = []
    gross_values = []
    turnover_values = []
    long_values = []
    short_values = []
    pnl_sum = 0.0
    cost_sum = 0.0
    for ret, pos in zip(returns.astype(np.float64), positions.astype(np.float64), strict=True):
        turnover = float(np.abs(pos - prev).sum())
        short_exp = float(np.abs(np.minimum(pos, 0.0)).sum())
        long_exp = float(np.maximum(pos, 0.0).sum())
        pnl = float((pos * ret).sum())
        cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
        net = pnl - cost
        net_values.append(net)
        gross_values.append(float(np.abs(pos).sum()))
        turnover_values.append(turnover)
        long_values.append(long_exp)
        short_values.append(short_exp)
        pnl_sum += pnl
        cost_sum += cost
        prev = pos
    net_arr = np.asarray(net_values, dtype=np.float64)
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + net_arr)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0 if len(equity) else np.array([0.0])
    total_return = float(equity[-1] / INITIAL_CAPITAL - 1.0) if len(equity) else 0.0
    vol = float(net_arr.std())
    sharpe = float(net_arr.mean() / vol * ANNUALIZATION) if vol > 0 else 0.0
    return {
        "strategy": name,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "win_rate_pct": round(float((net_arr > 0).mean()) * 100, 1) if len(net_arr) else 0.0,
        "position_bar_count": int((np.abs(positions) > 1e-12).sum()),
        "final_value": round(float(equity[-1]) if len(equity) else INITIAL_CAPITAL, 2),
        "avg_gross_exposure": round(float(np.mean(gross_values)) if gross_values else 0.0, 4),
        "avg_turnover": round(float(np.mean(turnover_values)) if turnover_values else 0.0, 4),
        "avg_long_exposure": round(float(np.mean(long_values)) if long_values else 0.0, 4),
        "avg_short_exposure": round(float(np.mean(short_values)) if short_values else 0.0, 4),
        "gross_pnl_sum": round(float(pnl_sum), 6),
        "cost_sum": round(float(cost_sum), 6),
        "net_pnl_sum": round(float(net_arr.sum()), 6),
        "daily_returns": [float(x) for x in net_arr],
        "equity_curve": [float(x) for x in equity],
    }


def score_metric(metrics: dict[str, object]) -> float:
    ret = float(metrics["total_return_pct"]) / 100.0
    sharpe = float(metrics["sharpe"])
    mdd = abs(float(metrics["max_drawdown_pct"]) / 100.0)
    turn = float(metrics["avg_turnover"])
    return sharpe + 0.4 * ret - 0.8 * mdd - 6.0 * turn


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def positions_to_frame(bundle: DataBundle, idx: np.ndarray, positions: np.ndarray, col: str) -> pd.DataFrame:
    rows = []
    for local_i, ti in enumerate(idx):
        ts = bundle.timestamps[int(ti)]
        for pi, pair in enumerate(bundle.pairs):
            rows.append({"timestamp": ts, "pair": pair, col: float(positions[local_i, pi])})
    return pd.DataFrame(rows)


def plot_results(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.25)
    plt.title("V15 Direct RL Strict Holdout Equity")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v15_direct_rl_equity.png", dpi=160)
    plt.close()


def write_report(output: dict[str, object]) -> None:
    lines = [
        "# V15 Direct RL Trader 报告",
        "",
        "## 1. 目标",
        "V15 按用户要求测试直接 RL 新模型：不使用 V12/V13/V14 预测信号，只用 4h 行情衍生特征和账户状态，policy 直接输出 15 个币的目标仓位。",
        "",
        "## 2. 时间切分",
        f"- Fit/train：{output['config']['fit_period']}",
        f"- Select：{output['config']['select_period']}",
        f"- Holdout：{output['config']['holdout_period']}",
        "",
        "## 3. 关键约束",
        "- State 禁用 `pred_z/confidence/edge/p_short/p_long/xgb_*` 等历史模型信号列。",
        "- 动作是 15 币目标仓位，经过单币 4% cap 和总 gross cap 投影。",
        "- Reward 内含手续费、滑点、funding、换手惩罚和风险惩罚。",
        "",
        "## 4. Holdout 结果",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 | 成本合计 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in output["results"]:
        lines.append(
            f"| {item['strategy']} | {item['total_return_pct']}% | {item['sharpe']} | {item['max_drawdown_pct']}% | "
            f"{item['avg_gross_exposure']} | {item['avg_turnover']} | {item['cost_sum']} |"
        )
    lines.extend([
        "",
        "## 5. 结论口径",
        "V15 是直接 RL/可微策略梯度原型。即使 holdout 为正，也只能说明该严格历史窗口上可交易；不能直接当成实盘证明，后续仍需 paper trading。",
    ])
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = require_cuda()
    print("[1/8] Loading data", flush=True)
    raw = load_input()
    bundle = standardize_bundle(make_bundle(raw))
    print(
        f"timestamps={len(bundle.timestamps)} pairs={len(bundle.pairs)} fit={len(bundle.fit_idx)} select={len(bundle.select_idx)} holdout={len(bundle.holdout_idx)}",
        flush=True,
    )

    print("[2/8] Training direct policies and selecting on 2024 select", flush=True)
    best = None
    best_score = -1e9
    selection_log = []
    for max_gross in MAX_GROSS_GRID:
        for temperature in TEMP_GRID:
            for turnover_penalty in TURNOVER_PENALTY_GRID:
                print(f"  candidate gross={max_gross} temp={temperature} turn_pen={turnover_penalty}", flush=True)
                model = train_one(bundle, device, max_gross, temperature, turnover_penalty, SEED + int(max_gross * 1000) + int(temperature * 100))
                select_pos = infer_positions(bundle, bundle.select_idx, model, device, max_gross, temperature)
                select_metrics = simulate_matrix(bundle.returns[bundle.select_idx], select_pos, "select")
                score = score_metric(select_metrics)
                row = {
                    "max_gross": max_gross,
                    "temperature": temperature,
                    "turnover_penalty": turnover_penalty,
                    "select_score": score,
                    "select_return_pct": select_metrics["total_return_pct"],
                    "select_sharpe": select_metrics["sharpe"],
                    "select_mdd_pct": select_metrics["max_drawdown_pct"],
                }
                selection_log.append(row)
                print(f"    select {row}", flush=True)
                if score > best_score:
                    best_score = score
                    best = (model, row)
    if best is None:
        raise RuntimeError("No V15 model selected")
    best_model, best_params = best
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_model.state_dict(), "params": best_params, "pair_features": PAIR_FEATURES, "market_features": MARKET_FEATURES}, CHECKPOINT_OUT)
    print(f"[3/8] Selected {best_params}", flush=True)

    print("[4/8] Applying frozen policy to 2025+ holdout", flush=True)
    holdout_pos = infer_positions(bundle, bundle.holdout_idx, best_model, device, float(best_params["max_gross"]), float(best_params["temperature"]))
    holdout_rl = simulate_matrix(bundle.returns[bundle.holdout_idx], holdout_pos, "V15-DirectRL-SelectedOn2024")

    print("[5/8] Building baselines", flush=True)
    flat_pos = np.zeros_like(holdout_pos)
    flat = simulate_matrix(bundle.returns[bundle.holdout_idx], flat_pos, "AlwaysFlat")
    ew_long = np.full_like(holdout_pos, 0.20 / len(bundle.pairs))
    ew = simulate_matrix(bundle.returns[bundle.holdout_idx], ew_long, "EqualWeightLong20")

    results = {
        "V15-DirectRL-SelectedOn2024": holdout_rl,
        "AlwaysFlat": flat,
        "EqualWeightLong20": ew,
    }
    ranked = sorted((strip_series(v) for v in results.values()), key=lambda x: float(x.get("sharpe", 0.0)), reverse=True)

    print("[6/8] Writing decisions", flush=True)
    hold_df = raw[raw["timestamp"].isin([bundle.timestamps[int(i)] for i in bundle.holdout_idx])].copy()
    pos_df = positions_to_frame(bundle, bundle.holdout_idx, holdout_pos, "pos_V15_DirectRL")
    decisions = hold_df.merge(pos_df, on=["timestamp", "pair"], how="left")
    decisions.to_parquet(DECISIONS_OUT, index=False)

    print("[7/8] Writing results/report", flush=True)
    output = {
        "methodology": "direct actor-only RL-style policy, no V12/V13 prediction columns in state, strict V14 split",
        "config": {
            "fit_period": f"{bundle.timestamps[int(bundle.fit_idx[0])]} to {bundle.timestamps[int(bundle.fit_idx[-1])]}",
            "select_period": f"{bundle.timestamps[int(bundle.select_idx[0])]} to {bundle.timestamps[int(bundle.select_idx[-1])]}",
            "holdout_period": f"{bundle.timestamps[int(bundle.holdout_idx[0])]} to {bundle.timestamps[int(bundle.holdout_idx[-1])]}",
            "pairs": bundle.pairs,
            "state_pair_features": PAIR_FEATURES,
            "state_market_features": MARKET_FEATURES,
            "forbidden_state_cols": sorted(FORBIDDEN_STATE_COLS),
            "selected_params": best_params,
            "selection_log": selection_log,
            "epochs": EPOCHS,
            "train_max_steps": TRAIN_MAX_STEPS,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "results": ranked,
    }
    RESULTS_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    BACKTEST_OUT.write_text(json.dumps({k: strip_series(v) for k, v in results.items()}, indent=2, ensure_ascii=False))
    plot_results(results)
    write_report(output)
    print("[8/8] Done", flush=True)
    for item in ranked:
        print(item, flush=True)


if __name__ == "__main__":
    main()
