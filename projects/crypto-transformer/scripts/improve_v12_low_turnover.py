#!/usr/bin/env python3
"""V12 low-turnover post-processing improvement.

This script does not retrain the Transformer. It reads the frozen V12 4h
predictions from data/v12_trade_decisions.parquet and adds conservative
low-turnover trading layers to diagnose whether V12 failed because of signal
quality or because 4h rebalancing costs overwhelmed the signal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

INPUT = DATA_DIR / "v12_trade_decisions.parquet"
RESULTS_JSON = DATA_DIR / "results_v12.json"
BACKTEST_JSON = DATA_DIR / "backtest_v12_results.json"
REPORT = DOCS_DIR / "v12_experiment_report.md"
IMPROVED_JSON = DATA_DIR / "results_v12_improved.json"
IMPROVED_BACKTEST_JSON = DATA_DIR / "backtest_v12_improved_results.json"

INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_PER_BAR = 0.0001 / 6.0
ANNUALIZATION = math.sqrt(365 * 6)


def load_decisions() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run scripts/train_v12.py first")
    df = pd.read_parquet(INPUT).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    required = {"timestamp", "pair", "next_1bar_ret", "pred_z", "confidence", "vol_24"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["score"] = df["pred_z"].astype(float) * df["confidence"].astype(float)
    return df


def low_turnover_positions(
    df: pd.DataFrame,
    *,
    confidence_threshold: float,
    quantile: float,
    max_gross: float,
    rebalance_bars: int,
) -> pd.Series:
    timestamps = pd.Index(sorted(df["timestamp"].unique()), name="timestamp")
    ts_rank = {ts: idx for idx, ts in enumerate(timestamps)}
    work = df.copy()
    work["tidx"] = work["timestamp"].map(ts_rank).astype(int)

    rebalance_rows = work[work["tidx"].mod(rebalance_bars).eq(0)].copy()
    rebalance_rows = rebalance_rows[rebalance_rows["confidence"] >= confidence_threshold]
    rebalance_rows["rank_pct"] = rebalance_rows.groupby("timestamp")["score"].rank(pct=True, method="first")
    rebalance_rows["raw_side"] = 0.0
    rebalance_rows.loc[rebalance_rows["rank_pct"] >= 1.0 - quantile, "raw_side"] = 1.0
    rebalance_rows.loc[rebalance_rows["rank_pct"] <= quantile, "raw_side"] = -1.0
    rebalance_rows = rebalance_rows[rebalance_rows["raw_side"].ne(0.0)].copy()
    if rebalance_rows.empty:
        return pd.Series(0.0, index=df.index)

    vol = rebalance_rows["vol_24"].replace(0.0, np.nan)
    rebalance_rows["raw_weight"] = rebalance_rows["raw_side"] * rebalance_rows["score"].abs() / vol
    rebalance_rows["raw_weight"] = rebalance_rows["raw_weight"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = rebalance_rows.groupby("timestamp")["raw_weight"].transform(lambda s: float(np.abs(s).sum()))
    rebalance_rows["position"] = np.where(gross > 0.0, rebalance_rows["raw_weight"] / gross * max_gross, 0.0)

    wide = (
        rebalance_rows.pivot(index="timestamp", columns="pair", values="position")
        .reindex(timestamps)
        .ffill()
        .fillna(0.0)
    )
    long_exp = wide.clip(lower=0.0).sum(axis=1)
    short_exp = (-wide.clip(upper=0.0)).sum(axis=1)
    gross_exp = long_exp + short_exp
    scale = np.ones(len(wide), dtype=float)
    gross_arr = gross_exp.to_numpy(dtype=float)
    np.divide(max_gross, gross_arr, out=scale, where=gross_arr > max_gross)
    wide = wide.mul(scale, axis=0)

    flat = wide.stack().rename("position").reset_index()
    merged = (
        df[["timestamp", "pair"]]
        .reset_index()
        .merge(flat, on=["timestamp", "pair"], how="left")
        .sort_values("index")
    )
    return merged["position"].fillna(0.0).set_axis(df.index)


def simulate(df: pd.DataFrame, position: pd.Series, name: str) -> dict[str, object]:
    work = df[["timestamp", "pair", "next_1bar_ret"]].copy()
    work["position"] = position.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    wide = work.pivot(index="timestamp", columns="pair", values="position").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = wide.iloc[0].abs().sum()

    pnl = (work["position"] * work["next_1bar_ret"]).groupby(work["timestamp"]).sum().reindex(wide.index).fillna(0.0)
    short_exp = work["position"].where(work["position"] < 0.0, 0.0).abs().groupby(work["timestamp"]).sum().reindex(wide.index).fillna(0.0)
    cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
    net = pnl - cost
    equity = INITIAL_CAPITAL * (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0) if len(equity) else 0.0
    vol = float(net.std(ddof=0))
    sharpe = float(net.mean() / vol * ANNUALIZATION) if vol > 0 else 0.0
    gross = wide.abs().sum(axis=1)
    long_exp = wide.clip(lower=0.0).sum(axis=1)

    return {
        "strategy": name,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100 if len(drawdown) else 0.0, 2),
        "win_rate_pct": round(float((net > 0).mean()) * 100 if len(net) else 0.0, 1),
        "trades": int((work["position"].abs() > 1e-12).sum()),
        "final_value": round(float(equity.iloc[-1]) if len(equity) else INITIAL_CAPITAL, 2),
        "avg_gross_exposure": round(float(gross.mean()) if len(gross) else 0.0, 4),
        "avg_turnover": round(float(turnover.mean()) if len(turnover) else 0.0, 4),
        "avg_long_exposure": round(float(long_exp.mean()) if len(long_exp) else 0.0, 4),
        "avg_short_exposure": round(float(short_exp.mean()) if len(short_exp) else 0.0, 4),
        "gross_pnl_sum": round(float(pnl.sum()), 6),
        "cost_sum": round(float(cost.sum()), 6),
        "net_pnl_sum": round(float(net.sum()), 6),
        "daily_returns": [float(x) for x in net.to_numpy()],
        "equity_curve": [float(x) for x in equity.to_numpy()],
    }


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_improved(all_results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in all_results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.4)
    plt.title("V12 Low-Turnover Improvement Equity Curves")
    plt.ylabel("Equity")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v12_low_turnover_equity.png", dpi=160)
    plt.close()


def main() -> None:
    df = load_decisions()

    # Conservative fixed candidate chosen after diagnosing that cost/turnover caused
    # the original V12 failure. This is an improvement experiment, not a fresh holdout.
    candidate = {
        "confidence_threshold": 0.30,
        "quantile": 0.20,
        "max_gross": 0.20,
        "rebalance_bars": 12,
    }
    exploratory_candidate = {
        "confidence_threshold": 0.30,
        "quantile": 0.20,
        "max_gross": 0.30,
        "rebalance_bars": 12,
    }

    pos_conservative = low_turnover_positions(df, **candidate)
    pos_exploratory = low_turnover_positions(df, **exploratory_candidate)
    df["pos_V12_LowTurnover"] = pos_conservative
    df["pos_V12_LowTurnover_Exploratory"] = pos_exploratory
    df.to_parquet(INPUT, index=False)

    existing_results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {"results": []}
    original_metrics = {item["strategy"]: item for item in existing_results.get("results", [])}
    improved = {
        "V12-LowTurnover": simulate(df, pos_conservative, "V12-LowTurnover"),
        "V12-LowTurnover-Exploratory": simulate(df, pos_exploratory, "V12-LowTurnover-Exploratory"),
    }
    all_for_plot: dict[str, dict[str, object]] = {}
    for col, name in [
        ("pos_V12_Model", "V12-Model"),
        ("pos_AlwaysFlat", "AlwaysFlat"),
    ]:
        if col in df.columns:
            all_for_plot[name] = simulate(df, df[col], name)
    all_for_plot.update(improved)
    plot_improved(all_for_plot)

    ranked = sorted([*original_metrics.values(), *[strip_series(v) for v in improved.values()]], key=lambda x: float(x.get("sharpe", 0.0)), reverse=True)
    output = {
        "best_strategy_by_sharpe": ranked[0]["strategy"] if ranked else None,
        "candidate_params": candidate,
        "exploratory_params": exploratory_candidate,
        "methodology_note": "Low-turnover variants use frozen V12 predictions and are post-training trading-layer improvements; they are not untouched holdout results.",
        "results": ranked,
    }
    IMPROVED_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    IMPROVED_BACKTEST_JSON.write_text(json.dumps({k: strip_series(v) for k, v in improved.items()}, indent=2, ensure_ascii=False))

    report = REPORT.read_text() if REPORT.exists() else "# V12 高频 4h 交易模型实验报告\n"
    block = f"""

## 8. 第二轮改进：低换手交易层

第一轮 V12 失败的主因不是单纯预测输出为随机，而是交易层每根 4h bar 大幅换仓，成本吞噬了全部毛收益：原始 `V12-Model` 平均总敞口约 0.995、平均每 bar 换手 0.794，累计毛 PnL 约 -0.477，但累计交易成本约 14.655，最终净值归零。

因此第二轮没有重新训练 Transformer，而是冻结 V12 预测，改为低换手交易层：每 12 根 4h bar（约 2 天）再平衡一次，只交易置信度超过 {candidate['confidence_threshold']} 的信号，做多/做空 score 排名前后 {candidate['quantile']:.0%}，最大总敞口限制为 {candidate['max_gross']:.0%}。

| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 | 成本合计 |
|---|---:|---:|---:|---:|---:|---:|
| V12-LowTurnover | {improved['V12-LowTurnover']['total_return_pct']}% | {improved['V12-LowTurnover']['sharpe']} | {improved['V12-LowTurnover']['max_drawdown_pct']}% | {improved['V12-LowTurnover']['avg_gross_exposure']} | {improved['V12-LowTurnover']['avg_turnover']} | {improved['V12-LowTurnover']['cost_sum']} |
| V12-LowTurnover-Exploratory | {improved['V12-LowTurnover-Exploratory']['total_return_pct']}% | {improved['V12-LowTurnover-Exploratory']['sharpe']} | {improved['V12-LowTurnover-Exploratory']['max_drawdown_pct']}% | {improved['V12-LowTurnover-Exploratory']['avg_gross_exposure']} | {improved['V12-LowTurnover-Exploratory']['avg_turnover']} | {improved['V12-LowTurnover-Exploratory']['cost_sum']} |

这个结果说明 4h 预测信号可能仍有一点可利用信息，但必须用低频再平衡、低敞口和严格成本控制。需要强调：低换手参数是在看到第一轮 V12 失败后设计的二阶段改进，并非全新未见 holdout，因此只能作为研究候选，不能视为已经验证的实盘策略。

新增文件：
- `data/results_v12_improved.json`
- `data/backtest_v12_improved_results.json`
- `docs/figures/v12_low_turnover_equity.png`
"""
    if "## 8. 第二轮改进：低换手交易层" in report:
        report = report.split("\n## 8. 第二轮改进：低换手交易层", 1)[0].rstrip() + block
    else:
        report = report.rstrip() + block
    REPORT.write_text(report)

    print("Saved", IMPROVED_JSON)
    print("Saved", IMPROVED_BACKTEST_JSON)
    print("Saved", REPORT)
    for name, metrics in improved.items():
        print(name, strip_series(metrics))


if __name__ == "__main__":
    main()
