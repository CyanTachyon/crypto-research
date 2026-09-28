#!/usr/bin/env python3
"""V13: 4h low-turnover ensemble + regime router + risk controls.

This experiment uses the V12 4h feature/prediction table as input, but does
not reuse the post-hoc V12 low-turnover parameters as the main strategy.  Each
walk-forward fold trains XGBoost models on past fit data, chooses trading-layer
parameters on a separate past selection window, then applies those frozen
choices to the future trade window.
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
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

INPUT = DATA_DIR / "v12_trade_decisions.parquet"
DECISIONS_OUT = DATA_DIR / "v13_trade_decisions.parquet"
RESULTS_OUT = DATA_DIR / "results_v13.json"
BACKTEST_OUT = DATA_DIR / "backtest_v13_results.json"
REPORT_OUT = DOCS_DIR / "v13_experiment_report.md"

INITIAL_CAPITAL = 10_000.0
BAR_PER_DAY = 6
ANNUALIZATION = math.sqrt(365 * BAR_PER_DAY)
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_PER_BAR = 0.0001 / BAR_PER_DAY

MIN_FIT_BARS = 1440
SELECT_BARS = 360
TRADE_BARS = 1440
PURGE_BARS = 132
MAX_FIT_ROWS = 60_000
PAIR_CAP = 0.04
STOP_VOL_MULT = 2.0
TAKE_VOL_MULT = 3.0
KILL_DD = -0.12
COOLDOWN_BARS = 36
SEED = 20260608

FEATURE_COLS = [
    "pair_id",
    "log_ret_1", "hl_range", "oc_ret", "volume_log", "volume_ratio_24",
    "ret_3", "ret_6", "ret_12", "ret_24", "ret_42", "ret_72",
    "vol_6", "vol_12", "vol_24", "vol_42", "vol_72",
    "range_6", "range_24", "price_pos_24", "price_pos_72",
    "mom_z_12", "mom_z_24", "vol_ratio_6_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "ret_rank_6", "ret_z_cs", "vol_rank_24",
    "btc_ret_6", "btc_ret_24", "btc_vol_24",
    "market_ret_6", "market_vol_24",
    "p_short", "p_flat", "p_long", "pred_z", "raw_conf", "edge", "confidence",
    "score",
]


@dataclass(frozen=True)
class Fold:
    fold: int
    fit_dates: list[pd.Timestamp]
    select_dates: list[pd.Timestamp]
    trade_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class RouterThresholds:
    vol_hi: float
    mom_abs: float
    btc_abs: float


def load_input() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run V12 first")
    df = pd.read_parquet(INPUT).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    required = {"timestamp", "pair", "next_1bar_ret", "target_z_6bar", "label_action", "vol_24", *FEATURE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["score"] = df["pred_z"].astype(float) * df["confidence"].astype(float)
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["target_z_6bar", "label_action", "next_1bar_ret"])


def make_folds(dates: list[pd.Timestamp]) -> list[Fold]:
    folds: list[Fold] = []
    start = MIN_FIT_BARS + PURGE_BARS + SELECT_BARS + PURGE_BARS
    fold_id = 1
    while start < len(dates):
        end = min(start + TRADE_BARS, len(dates))
        select_end = start - PURGE_BARS
        select_start = max(0, select_end - SELECT_BARS)
        fit_end = select_start - PURGE_BARS
        if fit_end < MIN_FIT_BARS:
            start = end
            continue
        fit_dates = dates[:fit_end]
        select_dates = dates[select_start:select_end]
        trade_dates = dates[start:end]
        if fit_dates and select_dates and trade_dates:
            folds.append(Fold(fold_id, fit_dates, select_dates, trade_dates))
            fold_id += 1
        start = end
    return folds


def xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.82,
        colsample_bytree=0.82,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        verbosity=0,
    )


def xgb_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.82,
        colsample_bytree=0.82,
        min_child_weight=8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        verbosity=0,
    )


def train_predict_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fit = df[df["timestamp"].isin(fold.fit_dates)].copy()
    select = df[df["timestamp"].isin(fold.select_dates)].copy()
    trade = df[df["timestamp"].isin(fold.trade_dates)].copy()

    if len(fit) > MAX_FIT_ROWS:
        fit = fit.tail(MAX_FIT_ROWS).copy()

    print(f"  train rows={len(fit)} select rows={len(select)} trade rows={len(trade)}", flush=True)
    x_fit = fit[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_reg = fit["target_z_6bar"].clip(-5.0, 5.0).to_numpy(dtype=np.float32)
    y_cls = fit["label_action"].astype(int).to_numpy()

    reg = xgb_regressor()
    clf = xgb_classifier()
    print("  fitting XGB regressor", flush=True)
    reg.fit(x_fit, y_reg)
    print("  fitting XGB classifier", flush=True)
    clf.fit(x_fit, y_cls)
    print("  predicting select/trade", flush=True)

    def add_predictions(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        x = out[FEATURE_COLS].to_numpy(dtype=np.float32)
        pred_reg = reg.predict(x)
        proba = clf.predict_proba(x)
        if proba.shape[1] != 3:
            fixed = np.zeros((len(out), 3), dtype=float)
            for idx, cls in enumerate(clf.classes_):
                fixed[:, int(cls)] = proba[:, idx]
            proba = fixed
        out["xgb_pred_z"] = pred_reg.astype(float)
        out["xgb_p_short"] = proba[:, 0].astype(float)
        out["xgb_p_flat"] = proba[:, 1].astype(float)
        out["xgb_p_long"] = proba[:, 2].astype(float)
        out["xgb_edge"] = out["xgb_p_long"] - out["xgb_p_short"]
        out["xgb_conf"] = np.maximum(out["xgb_p_long"], out["xgb_p_short"]) * (1.0 - out["xgb_p_flat"])
        out["rule_score"] = 0.55 * out["mom_z_24"] + 0.25 * out["ret_z_cs"] - 0.20 * (out["vol_rank_24"] - 0.5)
        return out

    thresholds = router_thresholds(fit)
    return add_predictions(select), add_predictions(trade), {
        "vol_hi": thresholds.vol_hi,
        "mom_abs": thresholds.mom_abs,
        "btc_abs": thresholds.btc_abs,
    }


def router_thresholds(fit: pd.DataFrame) -> RouterThresholds:
    return RouterThresholds(
        vol_hi=float(fit["market_vol_24"].quantile(0.75)),
        mom_abs=float(fit["market_ret_6"].abs().quantile(0.58)),
        btc_abs=float(fit["btc_ret_24"].abs().quantile(0.58)),
    )


def add_regime(df: pd.DataFrame, thresholds: RouterThresholds) -> pd.DataFrame:
    out = df.copy()
    high_vol = out["market_vol_24"] >= thresholds.vol_hi
    up = (out["market_ret_6"] > thresholds.mom_abs) | (out["btc_ret_24"] > thresholds.btc_abs)
    down = (out["market_ret_6"] < -thresholds.mom_abs) | (out["btc_ret_24"] < -thresholds.btc_abs)
    out["regime"] = "neutral"
    out.loc[high_vol & ~(up | down), "regime"] = "high_vol_chop"
    out.loc[up & ~down, "regime"] = "trend_up"
    out.loc[down & ~up, "regime"] = "trend_down"
    out.loc[up & down, "regime"] = "conflict"
    return out


def ensemble_score(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    xgb_w, v12_w, rule_w = weights
    return xgb_w * df["xgb_pred_z"] + v12_w * df["pred_z"] + rule_w * df["rule_score"]


def target_positions(
    df: pd.DataFrame,
    *,
    weights: tuple[float, float, float],
    confidence_threshold: float,
    quantile: float,
    max_gross: float,
    rebalance_bars: int,
    thresholds: RouterThresholds,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    work = add_regime(df, thresholds)
    timestamps = pd.Index(sorted(work["timestamp"].unique()), name="timestamp")
    ts_rank = {ts: idx for idx, ts in enumerate(timestamps)}
    work["tidx"] = work["timestamp"].map(ts_rank).astype(int)
    work["v13_score"] = ensemble_score(work, weights)
    work["v13_conf"] = 0.55 * work["xgb_conf"] + 0.45 * work["confidence"]

    reb = work[work["tidx"].mod(rebalance_bars).eq(0)].copy()
    reb = reb[reb["v13_conf"] >= confidence_threshold].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)
    reb["rank_pct"] = reb.groupby("timestamp")["v13_score"].rank(pct=True, method="first")
    reb["side"] = 0.0
    reb.loc[reb["rank_pct"] >= 1.0 - quantile, "side"] = 1.0
    reb.loc[reb["rank_pct"] <= quantile, "side"] = -1.0
    reb = reb[reb["side"].ne(0.0)].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)

    # Router: reduce gross in chop/conflict; tilt long/short according to regime.
    reb["gross_scale"] = 1.0
    reb.loc[reb["regime"].isin(["high_vol_chop", "conflict"]), "gross_scale"] = 0.45
    reb["side_scale"] = 1.0
    reb.loc[(reb["regime"] == "trend_up") & (reb["side"] < 0.0), "side_scale"] = 0.70
    reb.loc[(reb["regime"] == "trend_up") & (reb["side"] > 0.0), "side_scale"] = 1.10
    reb.loc[(reb["regime"] == "trend_down") & (reb["side"] > 0.0), "side_scale"] = 0.70
    reb.loc[(reb["regime"] == "trend_down") & (reb["side"] < 0.0), "side_scale"] = 1.10

    vol = reb["vol_24"].replace(0.0, np.nan)
    reb["raw_weight"] = reb["side"] * reb["v13_score"].abs() * reb["side_scale"] / vol
    reb["raw_weight"] = reb["raw_weight"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross = reb.groupby("timestamp")["raw_weight"].transform(lambda s: float(np.abs(s).sum()))
    reb["base_pos"] = np.where(gross > 0.0, reb["raw_weight"] / gross, 0.0)
    gross_scale = reb.groupby("timestamp")["gross_scale"].transform("min")
    reb["position"] = reb["base_pos"] * max_gross * gross_scale
    reb["position"] = reb["position"].clip(lower=-PAIR_CAP, upper=PAIR_CAP)

    wide = reb.pivot(index="timestamp", columns="pair", values="position").reindex(timestamps).ffill().fillna(0.0)
    gross_exp = wide.abs().sum(axis=1)
    scale = np.ones(len(wide), dtype=float)
    gross_arr = gross_exp.to_numpy(dtype=float)
    np.divide(max_gross, gross_arr, out=scale, where=gross_arr > max_gross)
    wide = wide.mul(scale, axis=0)
    flat = wide.stack().rename("position").reset_index()
    merged = df[["timestamp", "pair"]].reset_index().merge(flat, on=["timestamp", "pair"], how="left").sort_values("index")
    return merged["position"].fillna(0.0).set_axis(df.index)


def direction_clipped_return(ret: pd.Series, pos: pd.Series, vol: pd.Series) -> pd.Series:
    take = (TAKE_VOL_MULT * vol).clip(lower=0.002, upper=0.20)
    stop = (STOP_VOL_MULT * vol).clip(lower=0.002, upper=0.15)
    effective = ret.astype(float).copy()
    long_mask = pos > 0.0
    short_mask = pos < 0.0
    effective.loc[long_mask] = np.minimum(np.maximum(effective.loc[long_mask], -stop.loc[long_mask]), take.loc[long_mask])
    effective.loc[short_mask] = np.minimum(np.maximum(effective.loc[short_mask], -take.loc[short_mask]), stop.loc[short_mask])
    return effective


def apply_kill_switch(df: pd.DataFrame, target_pos: pd.Series) -> pd.Series:
    work = df[["timestamp", "pair", "next_1bar_ret", "vol_24"]].copy()
    work["target_pos"] = target_pos.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    target_wide = work.pivot(index="timestamp", columns="pair", values="target_pos").fillna(0.0).sort_index()
    ret_wide = work.pivot(index="timestamp", columns="pair", values="next_1bar_ret").fillna(0.0).reindex(target_wide.index)
    vol_wide = work.pivot(index="timestamp", columns="pair", values="vol_24").fillna(0.0).reindex(target_wide.index)

    actual_rows: list[pd.Series] = []
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    cooldown_left = 0
    prev = pd.Series(0.0, index=target_wide.columns)
    for ts in target_wide.index:
        target = target_wide.loc[ts].copy()
        if cooldown_left > 0:
            actual = pd.Series(0.0, index=target_wide.columns)
            cooldown_left -= 1
        else:
            actual = target
        actual_rows.append(actual)

        ret = ret_wide.loc[ts]
        vol = vol_wide.loc[ts]
        eff_ret = direction_clipped_return(ret, actual, vol)
        pnl = float((actual * eff_ret).sum())
        turnover = float((actual - prev).abs().sum())
        short_exp = float(actual.clip(upper=0.0).abs().sum())
        cost = (FEE_RATE + SLIPPAGE_RATE) * turnover + FUNDING_PER_BAR * short_exp
        equity *= 1.0 + pnl - cost
        peak = max(peak, equity)
        if peak > 0 and equity / peak - 1.0 <= KILL_DD:
            cooldown_left = COOLDOWN_BARS
        prev = actual

    actual_wide = pd.DataFrame(actual_rows, index=target_wide.index, columns=target_wide.columns)
    flat = actual_wide.stack().rename("position").reset_index()
    merged = df[["timestamp", "pair"]].reset_index().merge(flat, on=["timestamp", "pair"], how="left").sort_values("index")
    return merged["position"].fillna(0.0).set_axis(df.index)


def simulate(df: pd.DataFrame, position: pd.Series, name: str, *, risk_clip: bool = True) -> dict[str, object]:
    work = df[["timestamp", "pair", "next_1bar_ret", "vol_24"]].copy()
    work["position"] = position.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    if risk_clip:
        work["effective_ret"] = direction_clipped_return(work["next_1bar_ret"], work["position"], work["vol_24"])
    else:
        work["effective_ret"] = work["next_1bar_ret"].astype(float)
    wide = work.pivot(index="timestamp", columns="pair", values="position").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = wide.iloc[0].abs().sum()
    pnl = (work["position"] * work["effective_ret"]).groupby(work["timestamp"]).sum().reindex(wide.index).fillna(0.0)
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
        "position_bar_count": int((work["position"].abs() > 1e-12).sum()),
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


def score_metric(metrics: dict[str, object]) -> float:
    ret = float(metrics["total_return_pct"]) / 100.0
    sharpe = float(metrics["sharpe"])
    mdd = abs(float(metrics["max_drawdown_pct"]) / 100.0)
    turn = float(metrics["avg_turnover"])
    return sharpe + 0.4 * ret - 0.8 * mdd - 8.0 * turn


def select_params(select: pd.DataFrame, thresholds: RouterThresholds) -> dict[str, object]:
    weight_grid = [(1.0, 0.0, 0.0), (0.55, 0.25, 0.20)]
    conf_grid = [0.25, 0.35]
    q_grid = [0.20]
    gross_grid = [0.15, 0.20]
    rebalance_grid = [12]
    best: dict[str, object] | None = None
    best_score = -1e9
    for weights in weight_grid:
        for conf in conf_grid:
            for q in q_grid:
                for gross in gross_grid:
                    for rebalance in rebalance_grid:
                        target = target_positions(
                            select,
                            weights=weights,
                            confidence_threshold=conf,
                            quantile=q,
                            max_gross=gross,
                            rebalance_bars=rebalance,
                            thresholds=thresholds,
                        )
                        actual = apply_kill_switch(select, target)
                        metrics = simulate(select, actual, "select")
                        score = score_metric(metrics)
                        if score > best_score:
                            best_score = score
                            best = {
                                "weights": weights,
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
        raise RuntimeError("No V13 parameter candidate selected")
    return best


def strip_series(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_results(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.3)
    plt.title("V13 Equity Curves")
    plt.ylabel("Equity")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v13_equity_curves.png", dpi=160)
    plt.close()

    ranked = sorted(results.values(), key=lambda x: float(x.get("sharpe", 0.0)), reverse=True)
    names = [str(x["strategy"]) for x in ranked]
    sharpes = [float(x["sharpe"]) for x in ranked]
    rets = [float(x["total_return_pct"]) for x in ranked]
    x = np.arange(len(names))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.18, sharpes, width=0.36, label="Sharpe")
    plt.bar(x + 0.18, [r / 100.0 for r in rets], width=0.36, label="Return/100")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.title("V13 Strategy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "v13_strategy_comparison.png", dpi=160)
    plt.close()


def write_report(results: dict[str, dict[str, object]], fold_log: list[dict[str, object]]) -> None:
    ranked = sorted((strip_series(v) for v in results.values()), key=lambda x: float(x.get("sharpe", 0.0)), reverse=True)
    best = ranked[0]
    lines = [
        "# V13 4h 低换手 Ensemble + Regime Router 实验报告",
        "",
        "## 1. 实验目标",
        "V13 不再让 Transformer 直接满仓高频交易，而是把 V12 的 4h 冻结预测、XGBoost tabular 模型和规则因子合成弱信号，再用低换手、regime router 和风险控制转成仓位。",
        "",
        "## 2. 数据与切分",
        "输入为 `data/v12_trade_decisions.parquet`，周期为 4h，15 个主流币，覆盖 2020-11-08 至 2026-06-07。每个 fold 使用 fit -> purge -> select -> purge -> trade 的 walk-forward，参数只在过去 select 窗口选择，再应用到未来 trade 窗口。",
        "",
        "## 3. 策略设计",
        "- 信号层：XGBoost 回归预测 6-bar 风险调整收益，XGBoost 分类预测 short/flat/long 概率，融合 V12 Transformer 预测和动量/截面规则分数。",
        "- 交易层：每 12 或 18 根 4h bar 再平衡一次，只交易置信度足够高的截面前/后 15%~20%。",
        "- Regime router：上涨趋势偏多、下跌趋势偏空，高波动震荡/冲突 regime 降低总敞口。",
        f"- 风控层：单币仓位上限 {PAIR_CAP:.0%}，波动止损 {STOP_VOL_MULT}x vol，止盈 {TAKE_VOL_MULT}x vol，组合回撤超过 {abs(KILL_DD):.0%} 后冷却 {COOLDOWN_BARS} 根 4h bar。",
        "",
        "## 4. 结果排名",
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
        "## 5. Fold 参数选择",
        "| Fold | Trade 起止 | weights | conf | q | gross | rebalance | select Sharpe |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in fold_log:
        lines.append(
            f"| {row['fold']} | {row['trade_start']} → {row['trade_end']} | {row['weights']} | "
            f"{row['confidence_threshold']} | {row['quantile']} | {row['max_gross']} | {row['rebalance_bars']} | {row['select_sharpe']} |"
        )
    lines.extend([
        "",
        "## 6. 结论",
        f"本次 V13 表格中的最高 Sharpe 策略是 `{best['strategy']}`，收益 {best['total_return_pct']}%，Sharpe {best['sharpe']}，最大回撤 {best['max_drawdown_pct']}%。",
        "需要强调：V13 仍然是研究回测，不是实盘承诺。它比 V12 的改进点在于参数选择被放入 walk-forward select 窗口，并加入了低换手和风险约束；但因为它仍基于同一批历史 4h 数据开发，下一步必须做前瞻 paper trading 或新增未见数据验证。",
        "",
        "## 7. 输出文件",
        "- `data/results_v13.json`",
        "- `data/backtest_v13_results.json`",
        "- `data/v13_trade_decisions.parquet`",
        "- `docs/figures/v13_equity_curves.png`",
        "- `docs/figures/v13_strategy_comparison.png`",
    ])
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("[1/6] Loading V12 decision frame", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    df = load_input()
    dates = sorted(df["timestamp"].unique())
    folds = make_folds(dates)
    if not folds:
        raise RuntimeError("No V13 folds created")
    print(f"rows={len(df)} dates={len(dates)} folds={len(folds)} period={dates[0]} -> {dates[-1]}", flush=True)

    decisions: list[pd.DataFrame] = []
    fold_log: list[dict[str, object]] = []
    for fold in folds:
        print(
            f"\n[fold {fold.fold}/{len(folds)}] fit {fold.fit_dates[0]}->{fold.fit_dates[-1]} "
            f"select {fold.select_dates[0]}->{fold.select_dates[-1]} trade {fold.trade_dates[0]}->{fold.trade_dates[-1]}",
            flush=True,
        )
        select_pred, trade_pred, thr_dict = train_predict_fold(df, fold)
        thresholds = RouterThresholds(**thr_dict)
        params = select_params(select_pred, thresholds)
        print(f"  selected {params}", flush=True)
        target = target_positions(
            trade_pred,
            weights=tuple(params["weights"]),
            confidence_threshold=float(params["confidence_threshold"]),
            quantile=float(params["quantile"]),
            max_gross=float(params["max_gross"]),
            rebalance_bars=int(params["rebalance_bars"]),
            thresholds=thresholds,
        )
        actual = apply_kill_switch(trade_pred, target)
        trade_pred["pos_V13_Target"] = target
        trade_pred["pos_V13_EnsembleRouterRisk"] = actual
        trade_pred["fold_v13"] = fold.fold
        for key, value in params.items():
            trade_pred[f"v13_{key}"] = str(value) if key == "weights" else value
        decisions.append(trade_pred)
        fold_log.append({
            "fold": fold.fold,
            "trade_start": str(fold.trade_dates[0]),
            "trade_end": str(fold.trade_dates[-1]),
            **params,
        })

    print("\n[5/6] Simulating final strategies", flush=True)
    out = pd.concat(decisions, ignore_index=True).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    results: dict[str, dict[str, object]] = {
        "V13-EnsembleRouterRisk": simulate(out, out["pos_V13_EnsembleRouterRisk"], "V13-EnsembleRouterRisk"),
        "V13-TargetNoKill": simulate(out, out["pos_V13_Target"], "V13-TargetNoKill"),
        "AlwaysFlat": simulate(out, pd.Series(0.0, index=out.index), "AlwaysFlat"),
    }
    if "pos_V12_LowTurnover" in out.columns:
        results["V12-LowTurnover-same-period"] = simulate(out, out["pos_V12_LowTurnover"], "V12-LowTurnover-same-period")
    if "pos_V12_LowTurnover_Exploratory" in out.columns:
        results["V12-LowTurnover-Exploratory-same-period"] = simulate(out, out["pos_V12_LowTurnover_Exploratory"], "V12-LowTurnover-Exploratory-same-period")

    ranked = sorted((strip_series(v) for v in results.values()), key=lambda x: float(x.get("sharpe", 0.0)), reverse=True)
    output = {
        "best_strategy_by_sharpe": ranked[0]["strategy"],
        "config": {
            "input": str(INPUT),
            "rows": int(len(out)),
            "dates": int(out["timestamp"].nunique()),
            "period": f"{out['timestamp'].min()} to {out['timestamp'].max()}",
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "funding_per_bar": FUNDING_PER_BAR,
            "pair_cap": PAIR_CAP,
            "kill_dd": KILL_DD,
            "cooldown_bars": COOLDOWN_BARS,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "fold_log": fold_log,
        "results": ranked,
    }

    print("[6/6] Writing outputs", flush=True)
    out.to_parquet(DECISIONS_OUT, index=False)
    RESULTS_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    BACKTEST_OUT.write_text(json.dumps({k: strip_series(v) for k, v in results.items()}, indent=2, ensure_ascii=False))
    plot_results(results)
    write_report(results, fold_log)

    for item in ranked:
        print(item, flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
