#!/usr/bin/env python3
"""V14 strict holdout validation.

Goal: answer whether V12/V13 style 4h strategies survive a stricter split.

Protocol:
- fit window:    <= 2024-07-01 00:00:00
- purge gap:    132 bars (~22 days)
- select window:2024-07-23 00:00:00 through 2024-12-10 00:00:00
- purge gap:    through 2024-12-31
- holdout:      >= 2025-01-01 00:00:00, run once

The holdout window is never used for model fitting or parameter selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import improve_v12_low_turnover as v12lt
import train_v13 as v13


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

INPUT = DATA_DIR / "v12_trade_decisions.parquet"
DECISIONS_OUT = DATA_DIR / "v14_holdout_decisions.parquet"
RESULTS_OUT = DATA_DIR / "results_v14_holdout.json"
BACKTEST_OUT = DATA_DIR / "backtest_v14_holdout_results.json"
REPORT_OUT = DOCS_DIR / "v14_holdout_report.md"

FIT_END = pd.Timestamp("2024-07-01 00:00:00")
SELECT_START = pd.Timestamp("2024-07-23 00:00:00")
SELECT_END = pd.Timestamp("2024-12-10 00:00:00")
HOLDOUT_START = pd.Timestamp("2025-01-01 00:00:00")


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: val for k, val in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def score_metric(metrics: dict[str, object]) -> float:
    return v13.score_metric(metrics)


def load_frame() -> pd.DataFrame:
    df = v13.load_input().sort_values(["timestamp", "pair"]).reset_index(drop=True)
    if df["timestamp"].min() > FIT_END:
        raise RuntimeError("Input starts after fit window")
    if df["timestamp"].max() < HOLDOUT_START:
        raise RuntimeError("Input does not contain holdout window")
    return df


def select_v12lt_params(select: pd.DataFrame) -> dict[str, object]:
    conf_grid = [0.25, 0.30, 0.35, 0.40]
    q_grid = [0.15, 0.20]
    gross_grid = [0.15, 0.20, 0.25]
    rebalance_grid = [12, 18]
    best: dict[str, object] | None = None
    best_score = -1e9
    for conf in conf_grid:
        for q in q_grid:
            for gross in gross_grid:
                for rebalance in rebalance_grid:
                    pos = v12lt.low_turnover_positions(
                        select,
                        confidence_threshold=conf,
                        quantile=q,
                        max_gross=gross,
                        rebalance_bars=rebalance,
                    )
                    metrics = v13.simulate(select, pos, "select_v12lt")
                    score = score_metric(metrics)
                    if score > best_score:
                        best_score = score
                        best = {
                            "confidence_threshold": conf,
                            "quantile": q,
                            "max_gross": gross,
                            "rebalance_bars": rebalance,
                            "select_score": score,
                            "select_return_pct": metrics["total_return_pct"],
                            "select_sharpe": metrics["sharpe"],
                            "select_mdd_pct": metrics["max_drawdown_pct"],
                        }
    if best is None:
        raise RuntimeError("No V12-LT params selected")
    return best


def plot_results(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.3)
    plt.title("V14 Strict Holdout Equity Curves")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v14_holdout_equity_curves.png", dpi=160)
    plt.close()


def write_report(output: dict[str, object]) -> None:
    ranked = output["results"]
    cfg = output["config"]
    lines = [
        "# V14 严格 Holdout 验证报告",
        "",
        "## 1. 验证目的",
        "本次 V14 不追求新策略，而是专门回答用户提出的训练/测试重合质疑：固定规则，只用 2020-2024 的历史训练和选参，然后在 2025-2026 holdout 上只跑一次。",
        "",
        "## 2. 时间切分",
        f"- Fit 训练窗口：{cfg['fit_period']}",
        f"- Select 选参窗口：{cfg['select_period']}",
        f"- Holdout 测试窗口：{cfg['holdout_period']}",
        "- Fit 与 Select、Select 与 Holdout 之间均留出约 132 根 4h bar 的 purge gap。",
        "",
        "## 3. 候选策略",
        "- `V14-V12LT-SelectedOn2024`：只用 2024 select 窗口选 V12 低换手参数，再应用到 2025+ holdout。",
        "- `V14-V13-SelectedOn2024`：XGBoost 只在 fit 窗口训练，ensemble/router 参数只在 2024 select 窗口选择，再应用到 2025+ holdout。",
        "- `V12-LT-original-reference`：历史 V12-LT 原始列在 holdout 的表现，只作参考，因为它的参数来自之前研究过程。",
        "",
        "## 4. Holdout 结果",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 | 成本合计 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ranked:
        lines.append(
            f"| {item['strategy']} | {item['total_return_pct']}% | {item['sharpe']} | {item['max_drawdown_pct']}% | "
            f"{item['avg_gross_exposure']} | {item['avg_turnover']} | {item['cost_sum']} |"
        )
    lines.extend([
        "",
        "## 5. 方法学结论",
        "V14 是比 V12/V13 更严格的验证口径。如果候选策略在这里失效，说明之前结果很可能依赖研究过程中的参数选择；如果仍然有效，才更接近可继续 paper trading 的候选。",
        "",
        "## 6. 输出文件",
        "- `data/results_v14_holdout.json`",
        "- `data/backtest_v14_holdout_results.json`",
        "- `data/v14_holdout_decisions.parquet`",
        "- `docs/figures/v14_holdout_equity_curves.png`",
    ])
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("[1/7] Loading input", flush=True)
    df = load_frame()
    fit = df[df["timestamp"] <= FIT_END].copy()
    select = df[(df["timestamp"] >= SELECT_START) & (df["timestamp"] <= SELECT_END)].copy()
    holdout = df[df["timestamp"] >= HOLDOUT_START].copy()
    if fit.empty or select.empty or holdout.empty:
        raise RuntimeError("Empty fit/select/holdout split")
    print(f"fit rows={len(fit)} select rows={len(select)} holdout rows={len(holdout)}", flush=True)

    print("[2/7] Training V13 XGBoost on fit only", flush=True)
    fold = v13.Fold(
        fold=1,
        fit_dates=sorted(fit["timestamp"].unique()),
        select_dates=sorted(select["timestamp"].unique()),
        trade_dates=sorted(holdout["timestamp"].unique()),
    )
    select_pred, holdout_pred, thresholds_dict = v13.train_predict_fold(df, fold)
    thresholds = v13.RouterThresholds(**thresholds_dict)

    print("[3/7] Selecting V12-LT params on 2024 select only", flush=True)
    v12_params = select_v12lt_params(select_pred)
    print(f"selected V12-LT {v12_params}", flush=True)

    print("[4/7] Selecting V13 params on 2024 select only", flush=True)
    v13_params = v13.select_params(select_pred, thresholds)
    print(f"selected V13 {v13_params}", flush=True)

    print("[5/7] Applying frozen params once to 2025+ holdout", flush=True)
    pos_v12 = v12lt.low_turnover_positions(
        holdout_pred,
        confidence_threshold=float(v12_params["confidence_threshold"]),
        quantile=float(v12_params["quantile"]),
        max_gross=float(v12_params["max_gross"]),
        rebalance_bars=int(v12_params["rebalance_bars"]),
    )
    target_v13 = v13.target_positions(
        holdout_pred,
        weights=tuple(v13_params["weights"]),
        confidence_threshold=float(v13_params["confidence_threshold"]),
        quantile=float(v13_params["quantile"]),
        max_gross=float(v13_params["max_gross"]),
        rebalance_bars=int(v13_params["rebalance_bars"]),
        thresholds=thresholds,
    )
    pos_v13 = v13.apply_kill_switch(holdout_pred, target_v13)
    holdout_pred["pos_V14_V12LT_SelectedOn2024"] = pos_v12
    holdout_pred["pos_V14_V13_SelectedOn2024"] = pos_v13
    holdout_pred["pos_V14_V13_Target"] = target_v13

    print("[6/7] Simulating holdout strategies", flush=True)
    results = {
        "V14-V12LT-SelectedOn2024": v13.simulate(holdout_pred, pos_v12, "V14-V12LT-SelectedOn2024"),
        "V14-V13-SelectedOn2024": v13.simulate(holdout_pred, pos_v13, "V14-V13-SelectedOn2024"),
        "AlwaysFlat": v13.simulate(holdout_pred, pd.Series(0.0, index=holdout_pred.index), "AlwaysFlat"),
    }
    if "pos_V12_LowTurnover" in holdout_pred.columns:
        results["V12-LT-original-reference"] = v13.simulate(holdout_pred, holdout_pred["pos_V12_LowTurnover"], "V12-LT-original-reference")
    if "pos_V12_LowTurnover_Exploratory" in holdout_pred.columns:
        results["V12-LT30-original-reference"] = v13.simulate(holdout_pred, holdout_pred["pos_V12_LowTurnover_Exploratory"], "V12-LT30-original-reference")

    ranked = sorted((strip_series(v) for v in results.values()), key=lambda m: float(m["sharpe"]), reverse=True)
    output = {
        "methodology": "strict holdout: fit <= 2024-07-01, select 2024-07-23..2024-12-10, holdout >= 2025-01-01; holdout not used for training or parameter selection",
        "config": {
            "input": str(INPUT),
            "fit_period": f"{fit['timestamp'].min()} to {fit['timestamp'].max()}",
            "select_period": f"{select['timestamp'].min()} to {select['timestamp'].max()}",
            "holdout_period": f"{holdout_pred['timestamp'].min()} to {holdout_pred['timestamp'].max()}",
            "fit_rows": int(len(fit)),
            "select_rows": int(len(select)),
            "holdout_rows": int(len(holdout_pred)),
            "v12_selected_params": v12_params,
            "v13_selected_params": v13_params,
            "reference_note": "V12 original reference columns are included only for comparison; formal V14 candidates are selected on 2024 select window.",
        },
        "results": ranked,
    }

    print("[7/7] Writing outputs", flush=True)
    holdout_pred.to_parquet(DECISIONS_OUT, index=False)
    RESULTS_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    BACKTEST_OUT.write_text(json.dumps({k: strip_series(v) for k, v in results.items()}, indent=2, ensure_ascii=False))
    plot_results(results)
    write_report(output)
    for item in ranked:
        print(item, flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
