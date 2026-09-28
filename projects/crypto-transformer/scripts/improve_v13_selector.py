#!/usr/bin/env python3
"""V13 post-run selector diagnostics.

This script does not retrain models. It reads V13 trade decisions and tests a
simple ensemble selector: per fold, choose the already-generated V13 position or
the V12 low-turnover baseline based on realized fold performance diagnostics.

Because this selection uses trade-period diagnostics, it is explicitly marked as
post-hoc and is only used to understand whether V13 failed from signal quality
or from model selection. It must not be treated as a validated live strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_v13 as v13


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

INPUT = DATA_DIR / "v13_trade_decisions.parquet"
RESULTS = DATA_DIR / "results_v13.json"
IMPROVED = DATA_DIR / "results_v13_improved.json"
BACKTEST = DATA_DIR / "backtest_v13_improved_results.json"
REPORT = DOCS_DIR / "v13_experiment_report.md"


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: val for k, val in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def main() -> None:
    df = pd.read_parquet(INPUT).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    required = {"fold_v13", "pos_V13_EnsembleRouterRisk", "pos_V12_LowTurnover", "pos_V12_LowTurnover_Exploratory"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selector = pd.Series(0.0, index=df.index)
    selector_cons = pd.Series(0.0, index=df.index)
    fold_choices: list[dict[str, object]] = []
    for fold, part in df.groupby("fold_v13", sort=True):
        idx = part.index
        candidates = {
            "V13": part["pos_V13_EnsembleRouterRisk"],
            "V12-LT20": part["pos_V12_LowTurnover"],
            "V12-LT30": part["pos_V12_LowTurnover_Exploratory"],
        }
        metrics = {name: v13.simulate(part, pos, name) for name, pos in candidates.items()}
        ranked = sorted(metrics.values(), key=lambda m: v13.score_metric(m), reverse=True)
        best_name = str(ranked[0]["strategy"])
        selector.loc[idx] = candidates[best_name].to_numpy(dtype=float)
        # Conservative selector may only choose V13 or 20% gross V12-LT.
        cons_names = ["V13", "V12-LT20"]
        cons_ranked = sorted((metrics[name] for name in cons_names), key=lambda m: v13.score_metric(m), reverse=True)
        cons_name = str(cons_ranked[0]["strategy"])
        selector_cons.loc[idx] = candidates[cons_name].to_numpy(dtype=float)
        fold_choices.append({
            "fold": int(fold),
            "posthoc_best": best_name,
            "conservative_best": cons_name,
            "v13_score": round(v13.score_metric(metrics["V13"]), 6),
            "v12_lt20_score": round(v13.score_metric(metrics["V12-LT20"]), 6),
            "v12_lt30_score": round(v13.score_metric(metrics["V12-LT30"]), 6),
            "v13_return_pct": metrics["V13"]["total_return_pct"],
            "v12_lt20_return_pct": metrics["V12-LT20"]["total_return_pct"],
            "v12_lt30_return_pct": metrics["V12-LT30"]["total_return_pct"],
        })

    df["pos_V13_PostHocSelector"] = selector
    df["pos_V13_ConservativeSelector"] = selector_cons
    df.to_parquet(INPUT, index=False)

    all_metrics = {
        "V13-EnsembleRouterRisk": v13.simulate(df, df["pos_V13_EnsembleRouterRisk"], "V13-EnsembleRouterRisk"),
        "V12-LowTurnover-same-period": v13.simulate(df, df["pos_V12_LowTurnover"], "V12-LowTurnover-same-period"),
        "V12-LowTurnover-Exploratory-same-period": v13.simulate(df, df["pos_V12_LowTurnover_Exploratory"], "V12-LowTurnover-Exploratory-same-period"),
        "V13-ConservativeSelector-posthoc": v13.simulate(df, df["pos_V13_ConservativeSelector"], "V13-ConservativeSelector-posthoc"),
        "V13-PostHocSelector": v13.simulate(df, df["pos_V13_PostHocSelector"], "V13-PostHocSelector"),
        "AlwaysFlat": v13.simulate(df, pd.Series(0.0, index=df.index), "AlwaysFlat"),
    }
    ranked = sorted((strip_series(m) for m in all_metrics.values()), key=lambda m: float(m["sharpe"]), reverse=True)
    output = {
        "methodology_note": "Selectors are post-hoc diagnostics using trade-fold outcomes. They are not validated live strategies.",
        "fold_choices": fold_choices,
        "results": ranked,
    }
    IMPROVED.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    BACKTEST.write_text(json.dumps({k: strip_series(v) for k, v in all_metrics.items()}, indent=2, ensure_ascii=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    v13.plot_results(all_metrics)

    base_report = REPORT.read_text() if REPORT.exists() else "# V13 实验报告\n"
    block = [
        "",
        "## 8. 第二轮诊断：模型/基线选择器",
        "",
        "第一轮 V13 的新 ensemble/router 风控策略为正收益，但没有超过 V12 低换手基线。第二轮没有重新训练，而是做诊断：如果每个 fold 允许在 V13 与 V12 低换手之间选择，收益是否来自 V13 新信号，还是仍主要来自 V12 的低换手交易层。",
        "",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in ranked:
        block.append(
            f"| {item['strategy']} | {item['total_return_pct']}% | {item['sharpe']} | {item['max_drawdown_pct']}% | {item['avg_gross_exposure']} | {item['avg_turnover']} |"
        )
    block.extend([
        "",
        "注意：`V13-PostHocSelector` 和 `V13-ConservativeSelector-posthoc` 使用了 trade fold 的事后表现来选择策略，只能作为失败分析，不能视为可实盘策略。若它们仍主要选择 V12-LT，说明 V13 新信号没有稳定增加 alpha。",
    ])
    if "## 8. 第二轮诊断：模型/基线选择器" in base_report:
        base_report = base_report.split("\n## 8. 第二轮诊断：模型/基线选择器", 1)[0].rstrip()
    REPORT.write_text(base_report.rstrip() + "\n" + "\n".join(block) + "\n")

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
