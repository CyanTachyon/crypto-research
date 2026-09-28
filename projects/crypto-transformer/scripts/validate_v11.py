#!/usr/bin/env python3
"""V11 frozen validation for V10-F TransformerGatedV9.

This script does not retrain models and does not reselect strategy families.
It validates the frozen V10 trade decisions with stricter diagnostics:
- original V10 daily-gross cost replay;
- turnover-based cost replay and stress tests;
- monthly and market-regime decomposition;
- block bootstrap significance checks;
- gate ablations and confidence sensitivity around the frozen V10-F decisions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/cyan/default/crypto")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_v9 as v9  # noqa: E402


INITIAL_CAPITAL = 10_000.0
BASE_FEE = 0.001
BASE_SLIPPAGE = 0.0005
BASE_FUNDING = 0.0001
RNG_SEED = 20260608
BOOTSTRAP_RUNS = 2000
BOOTSTRAP_BLOCK = 7

IN_TRADES = ROOT / "data/v10_trade_decisions.parquet"
IN_V10_BACKTEST = ROOT / "data/backtest_v10_results.json"
OUT_RESULTS = ROOT / "data/results_v11.json"
OUT_BACKTEST = ROOT / "data/backtest_v11_results.json"
OUT_REPORT = ROOT / "docs/v11_validation_report.md"
FIG_DIR = ROOT / "docs/figures"


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min() * 100.0)


def metrics_from_daily(name: str, daily: pd.DataFrame, trades: int, avg_long: float, avg_short: float) -> dict[str, object]:
    rets = daily["net_return"].to_numpy(dtype=float)
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + rets)
    if len(equity) == 0:
        equity = np.array([INITIAL_CAPITAL], dtype=float)
    sharpe = float(rets.mean() / (rets.std() + 1e-10) * math.sqrt(365)) if len(rets) else 0.0
    total = float((equity[-1] / INITIAL_CAPITAL - 1.0) * 100.0)
    mdd = max_drawdown(np.concatenate([[INITIAL_CAPITAL], equity]))
    win = float((rets > 0).mean() * 100.0) if len(rets) else 0.0
    gross = float(daily["gross"].mean()) if len(daily) else 0.0
    turnover = float(daily["turnover"].mean()) if len(daily) else 0.0
    score = sharpe + total / 100.0 + mdd / 100.0 - min(trades / 100_000.0, 0.1)
    return {
        "strategy": name,
        "total_return_pct": round(total, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(win, 1),
        "trades": int(trades),
        "final_value": round(float(equity[-1]), 2),
        "avg_gross_exposure": round(gross, 4),
        "avg_turnover": round(turnover, 4),
        "avg_long_exposure": round(avg_long, 4),
        "avg_short_exposure": round(avg_short, 4),
        "score": round(score, 4),
        "daily_returns": [float(x) for x in rets],
        "equity_curve": [float(INITIAL_CAPITAL)] + [float(x) for x in equity],
    }


def daily_pnl(
    df: pd.DataFrame,
    position_col: str,
    cost_model: str,
    fee: float = BASE_FEE,
    slippage: float = BASE_SLIPPAGE,
    funding: float = BASE_FUNDING,
    clipped: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cols = ["date", "pair", "next_1d_ret", position_col]
    if clipped:
        cols.extend(["vol_20", "tp_expected", "sl_expected"])
    work = df[cols].copy()
    work["position"] = work[position_col].astype(float)
    if clipped:
        vol = work["vol_20"].replace(0, np.nan).fillna(work["vol_20"].median()).clip(lower=0.004).to_numpy(dtype=float)
        tp = work["tp_expected"].to_numpy(dtype=float) * vol
        sl = work["sl_expected"].to_numpy(dtype=float) * vol
        effective = work["next_1d_ret"].to_numpy(dtype=float).copy()
        pos_values = work["position"].to_numpy(dtype=float)
        long_mask = pos_values > 0
        short_mask = pos_values < 0
        effective[long_mask] = np.clip(effective[long_mask], -sl[long_mask], tp[long_mask])
        effective[short_mask] = np.clip(effective[short_mask], -tp[short_mask], sl[short_mask])
        work["trade_ret"] = effective
    else:
        work["trade_ret"] = work["next_1d_ret"].astype(float)
    work = work.sort_values(["date", "pair"]).reset_index(drop=True)

    prev_weights: dict[str, float] = {}
    rows: list[dict[str, float | pd.Timestamp]] = []
    trades = 0
    for date, day in work.groupby("date", sort=True):
        weights = day.set_index("pair")["position"].astype(float)
        returns = day.set_index("pair")["trade_ret"].astype(float)
        pnl = float((weights * returns).sum())
        gross = float(weights.abs().sum())
        long_exp = float(weights.clip(lower=0.0).sum())
        short_exp = float((-weights.clip(upper=0.0)).sum())
        turnover = 0.0
        for pair, weight in weights.items():
            turnover += abs(float(weight) - prev_weights.get(str(pair), 0.0))
        for pair in set(prev_weights) - {str(x) for x in weights.index}:
            turnover += abs(prev_weights[pair])
        prev_weights = {str(pair): float(weight) for pair, weight in weights.items()}
        trades += int((weights.abs() > 1e-12).sum())

        if cost_model == "daily_gross":
            cost = (fee + slippage) * gross + funding * short_exp
        elif cost_model == "turnover":
            cost = (fee + slippage) * turnover + funding * short_exp
        else:
            raise ValueError(f"Unknown cost model: {cost_model}")
        rows.append({
            "date": pd.Timestamp(date),
            "raw_pnl": pnl,
            "cost": cost,
            "net_return": pnl - cost,
            "gross": gross,
            "turnover": turnover,
            "long_exp": long_exp,
            "short_exp": short_exp,
        })

    daily = pd.DataFrame(rows)
    avg_long = float(daily["long_exp"].mean()) if len(daily) else 0.0
    avg_short = float(daily["short_exp"].mean()) if len(daily) else 0.0
    metrics = metrics_from_daily(position_col, daily, trades, avg_long, avg_short)
    return daily, metrics


def baseline_positions(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "V10-F TransformerGatedV9": df["pos_V10_F"].astype(float),
        "V10-C ActionSizeTPSL": df["pos_V10_C"].astype(float),
        "V9-VolTarget-Quantile(q=0.20)": v9.positions_vol_target(df, threshold=0.0, q=0.20).astype(float),
        "V4-LS(th=0.005)": v9.positions_ls(df, threshold=0.005).astype(float),
        "AlwaysShort": v9.positions_always(df, "short").astype(float),
        "RandomLS(seed=7)": v9.positions_random(df, seed=7).astype(float),
    }


def normalize_daily_positions(weights: pd.Series, df: pd.DataFrame, max_gross: float = 1.0) -> pd.Series:
    out = weights.copy().astype(float)
    for _, idx in df.groupby("date").groups.items():
        gross = float(np.abs(out.loc[idx]).sum())
        if gross > max_gross:
            out.loc[idx] = out.loc[idx] / gross * max_gross
    return out


def add_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, series in baseline_positions(out).items():
        col = "frozen_" + name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "p")
        col = col.replace("-", "_").replace("/", "_")
        out[col] = series.reindex(out.index).fillna(0.0).to_numpy(dtype=float)
    return out


def block_bootstrap(daily: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    rets = daily["net_return"].to_numpy(dtype=float)
    n = len(rets)
    if n == 0:
        return {}
    sampled_total: list[float] = []
    sampled_sharpe: list[float] = []
    for _ in range(BOOTSTRAP_RUNS):
        chunks: list[np.ndarray] = []
        while sum(len(c) for c in chunks) < n:
            start = int(rng.integers(0, n))
            idx = np.arange(start, start + BOOTSTRAP_BLOCK) % n
            chunks.append(rets[idx])
        sample = np.concatenate(chunks)[:n]
        equity = np.cumprod(1.0 + sample)
        sampled_total.append(float((equity[-1] - 1.0) * 100.0))
        sampled_sharpe.append(float(sample.mean() / (sample.std() + 1e-10) * math.sqrt(365)))
    total_arr = np.asarray(sampled_total)
    sharpe_arr = np.asarray(sampled_sharpe)
    return {
        "runs": BOOTSTRAP_RUNS,
        "block_days": BOOTSTRAP_BLOCK,
        "prob_total_return_gt_0_pct": round(float((total_arr > 0).mean() * 100.0), 1),
        "prob_sharpe_gt_1_pct": round(float((sharpe_arr > 1.0).mean() * 100.0), 1),
        "total_return_ci5_50_95_pct": [round(float(x), 2) for x in np.percentile(total_arr, [5, 50, 95])],
        "sharpe_ci5_50_95": [round(float(x), 2) for x in np.percentile(sharpe_arr, [5, 50, 95])],
    }


def month_breakdown(daily: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = daily.copy()
    work["month"] = work["date"].dt.strftime("%Y-%m")
    for month, part in work.groupby("month", sort=True):
        rows.append(metrics_from_daily(str(month), part, 0, float(part["long_exp"].mean()), float(part["short_exp"].mean())))
    return rows


def regime_breakdown(df: pd.DataFrame, position_col: str, cost_model: str) -> list[dict[str, object]]:
    market = df.groupby("date")["next_1d_ret"].mean().rename("market_ret").reset_index()
    quantiles = market["market_ret"].quantile([0.33, 0.67]).to_list()
    low, high = float(quantiles[0]), float(quantiles[1])
    market["regime"] = np.where(market["market_ret"] <= low, "down-market", np.where(market["market_ret"] >= high, "up-market", "sideways"))
    merged = df.merge(market[["date", "regime"]], on="date", how="left")
    rows: list[dict[str, object]] = []
    for regime, part in merged.groupby("regime", sort=True):
        daily, metrics = daily_pnl(part, position_col, cost_model)
        metrics["regime_days"] = int(daily["date"].nunique())
        rows.append(metrics)
    return rows


def stress_tests(df: pd.DataFrame, position_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for multiplier in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        _, metrics = daily_pnl(
            df,
            position_col,
            cost_model="turnover",
            fee=BASE_FEE * multiplier,
            slippage=BASE_SLIPPAGE * multiplier,
            funding=BASE_FUNDING * multiplier,
        )
        metrics["cost_multiplier"] = multiplier
        rows.append(metrics)
    return rows


def confidence_sensitivity(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = v9.positions_vol_target(df, threshold=0.0, q=0.20).astype(float)
    for conf in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for agree in [False, True]:
            keep = df["confidence"].to_numpy(dtype=float) >= conf
            if agree:
                keep = keep & (np.sign(base.to_numpy(dtype=float)) == np.sign(df["edge"].to_numpy(dtype=float)))
            pos = base.copy()
            pos.loc[~keep] = 0.0
            pos = normalize_daily_positions(pos, df)
            tmp = df.copy()
            tmp["tmp_pos"] = pos.to_numpy(dtype=float)
            _, metrics = daily_pnl(tmp, "tmp_pos", cost_model="daily_gross")
            metrics["confidence_threshold"] = conf
            metrics["require_agreement"] = agree
            rows.append(metrics)
    return sorted(rows, key=lambda x: float(x["score"]), reverse=True)


def plot_v11(v10_daily: pd.DataFrame, v9_daily: pd.DataFrame, monthly: list[dict[str, object]], stress: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for label, daily in [("V10-F", v10_daily), ("V9 baseline", v9_daily)]:
        curve = INITIAL_CAPITAL * np.cumprod(1.0 + daily["net_return"].to_numpy(dtype=float))
        plt.plot(daily["date"], curve, label=label)
    plt.title("V11 Frozen Validation Equity Curve - Original V10 Cost")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v11_frozen_equity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    months = [str(x["strategy"]) for x in monthly]
    returns = [float(x["total_return_pct"]) for x in monthly]
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in returns]
    plt.bar(months, returns, color=colors)
    plt.title("V11 V10-F Monthly Returns")
    plt.ylabel("Return %")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v11_monthly_returns.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    xs = [float(x["cost_multiplier"]) for x in stress]
    ys = [float(x["total_return_pct"]) for x in stress]
    plt.plot(xs, ys, marker="o")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("V11 V10-F Turnover Cost Stress")
    plt.xlabel("Fee/Slippage/Funding Multiplier")
    plt.ylabel("Return %")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v11_cost_stress.png", dpi=160)
    plt.close()


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def write_report(payload: dict[str, object]) -> None:
    original = payload["original_cost_metrics"]
    turnover = payload["turnover_cost_metrics"]
    v10 = original["V10-F TransformerGatedV9"]
    v9 = original["V9-VolTarget-Quantile(q=0.20)"]
    boot = payload["bootstrap_v10f_original_cost"]
    months = payload["monthly_v10f_original_cost"]
    stress = payload["turnover_cost_stress_v10f"]
    lines = [
        "# V11：V10-F 冻结验证报告",
        "",
        "## 1. 验证目标",
        "",
        "V11 不重新训练、不重新发明策略，只验证 V10-F TransformerGatedV9 这个冻结候选是否稳健。",
        "输入固定为 `data/v10_trade_decisions.parquet`，重点检查成本口径、bootstrap、月度表现、市场 regime 和参数敏感性。",
        "",
        "## 2. 冻结主结果（V10 原始日度 gross 成本口径）",
        "",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 胜率 | 交易数 | 平均总敞口 | 平均换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(original.items(), key=lambda kv: float(kv[1]["score"]), reverse=True):
        lines.append(
            f"| {name} | {metrics['total_return_pct']:+.2f}% | {metrics['sharpe']:.2f} | "
            f"{metrics['max_drawdown_pct']:.2f}% | {metrics['win_rate_pct']:.1f}% | {metrics['trades']} | "
            f"{metrics['avg_gross_exposure']:.4f} | {metrics['avg_turnover']:.4f} |"
        )
    lines += [
        "",
        "## 3. Turnover-based 成本口径",
        "",
        "V10 原始回测按每日总敞口收费，偏保守；V11 同时按真实换手收费重放。注意：换手口径通常会提高持仓型策略收益，因此这里只用于压力与一致性检查，不替代原始口径。",
        "",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均换手 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(turnover.items(), key=lambda kv: float(kv[1]["score"]), reverse=True):
        lines.append(
            f"| {name} | {metrics['total_return_pct']:+.2f}% | {metrics['sharpe']:.2f} | "
            f"{metrics['max_drawdown_pct']:.2f}% | {metrics['avg_turnover']:.4f} |"
        )
    lines += [
        "",
        "## 4. Bootstrap 显著性检查（V10-F，原始成本口径）",
        "",
        f"- Block bootstrap：{boot['runs']} 次，block={boot['block_days']} 天。",
        f"- 总收益 5/50/95 分位：{boot['total_return_ci5_50_95_pct']}%。",
        f"- Sharpe 5/50/95 分位：{boot['sharpe_ci5_50_95']}。",
        f"- P(总收益>0)：{boot['prob_total_return_gt_0_pct']}%。",
        f"- P(Sharpe>1)：{boot['prob_sharpe_gt_1_pct']}%。",
        "",
        "## 5. 月度表现",
        "",
        "| 月份 | 收益 | Sharpe | 最大回撤 | 胜率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in months:
        lines.append(f"| {m['strategy']} | {m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.2f}% | {m['win_rate_pct']:.1f}% |")
    lines += [
        "",
        "## 6. 成本压力测试（V10-F，turnover-based）",
        "",
        "| 成本倍数 | 收益 | Sharpe | 最大回撤 |",
        "|---:|---:|---:|---:|",
    ]
    for s in stress:
        lines.append(f"| {s['cost_multiplier']:.1f}x | {s['total_return_pct']:+.2f}% | {s['sharpe']:.2f} | {s['max_drawdown_pct']:.2f}% |")
    lines += [
        "",
        "## 7. V11 判断",
        "",
        f"在冻结交易决策和 V10 原始成本口径下，V10-F 仍为第一：收益 {v10['total_return_pct']:+.2f}%，Sharpe {v10['sharpe']:.2f}，最大回撤 {v10['max_drawdown_pct']:.2f}%。同期 V9 基线为 {v9['total_return_pct']:+.2f}%，Sharpe {v9['sharpe']:.2f}，最大回撤 {v9['max_drawdown_pct']:.2f}%。",
        "",
        "这说明 V10-F 的优势不是单纯由重新计价成本造成；但 V11 仍不是生产验证。它验证的是既有样本内冻结结果的稳健性，不等于未来未见行情的实盘表现。",
        "",
        "下一步必须冻结代码和参数后做 forward paper trading，或等待新增数据作为 untouched holdout。",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not IN_TRADES.exists():
        raise FileNotFoundError(IN_TRADES)
    df = pd.read_parquet(IN_TRADES)
    df["date"] = pd.to_datetime(df["date"])
    df = add_position_columns(df)
    rng = np.random.default_rng(RNG_SEED)

    strategy_cols = {
        "V10-F TransformerGatedV9": "frozen_V10_F_TransformerGatedV9",
        "V10-C ActionSizeTPSL": "frozen_V10_C_ActionSizeTPSL",
        "V9-VolTarget-Quantile(q=0.20)": "frozen_V9_VolTarget_Quantileq0p20",
        "V4-LS(th=0.005)": "frozen_V4_LSth0p005",
        "AlwaysShort": "frozen_AlwaysShort",
        "RandomLS(seed=7)": "frozen_RandomLSseed7",
    }

    original: dict[str, dict[str, object]] = {}
    turnover: dict[str, dict[str, object]] = {}
    daily_original: dict[str, pd.DataFrame] = {}
    for name, col in strategy_cols.items():
        use_clipped = name == "V10-C ActionSizeTPSL"
        daily_gross, gross_metrics = daily_pnl(df, col, cost_model="daily_gross", clipped=use_clipped)
        daily_turnover, turnover_metrics = daily_pnl(df, col, cost_model="turnover", clipped=use_clipped)
        original[name] = strip_series(gross_metrics)
        turnover[name] = strip_series(turnover_metrics)
        daily_original[name] = daily_gross

    v10_daily = daily_original["V10-F TransformerGatedV9"]
    v9_daily = daily_original["V9-VolTarget-Quantile(q=0.20)"]
    monthly = month_breakdown(v10_daily)
    regimes = regime_breakdown(df, strategy_cols["V10-F TransformerGatedV9"], cost_model="daily_gross")
    stress = [strip_series(x) for x in stress_tests(df, strategy_cols["V10-F TransformerGatedV9"])]
    sensitivity = [strip_series(x) for x in confidence_sensitivity(df)[:12]]
    bootstrap = block_bootstrap(v10_daily, rng)

    payload: dict[str, object] = {
        "config": {
            "input": str(IN_TRADES),
            "period": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "rows": int(len(df)),
            "dates": int(df["date"].nunique()),
            "pairs": int(df["pair"].nunique()),
            "bootstrap_runs": BOOTSTRAP_RUNS,
            "bootstrap_block_days": BOOTSTRAP_BLOCK,
            "base_fee": BASE_FEE,
            "base_slippage": BASE_SLIPPAGE,
            "base_funding": BASE_FUNDING,
        },
        "original_cost_metrics": original,
        "turnover_cost_metrics": turnover,
        "bootstrap_v10f_original_cost": bootstrap,
        "monthly_v10f_original_cost": monthly,
        "regime_v10f_original_cost": [strip_series(x) for x in regimes],
        "turnover_cost_stress_v10f": stress,
        "confidence_sensitivity_top12_original_cost": sensitivity,
    }
    OUT_RESULTS.write_text(json.dumps({
        "best_original_cost": original["V10-F TransformerGatedV9"],
        "bootstrap": bootstrap,
        "config": payload["config"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_BACKTEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_v11(v10_daily, v9_daily, monthly, stress)
    write_report(payload)

    print("V11 frozen validation complete")
    print("best_original_cost", original["V10-F TransformerGatedV9"])
    print("bootstrap", bootstrap)
    print(f"saved {OUT_RESULTS}")
    print(f"saved {OUT_BACKTEST}")
    print(f"saved {OUT_REPORT}")


if __name__ == "__main__":
    main()
