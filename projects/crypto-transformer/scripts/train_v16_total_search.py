#!/usr/bin/env python3
"""V16 total search: aggressive 4h strategy search with strict holdout.

This script searches many signal/trading-layer combinations on a fixed
pre-2025 selection window and applies only the selected candidates to the
2025+ holdout.  It is deliberately honest about two families:

- strict_raw: models/rules using only raw/derived 4h market features;
- augmented_v12: candidates that may use frozen V12 prediction columns.

The formal comparison must keep these families separate because V12 features
come from earlier research.  All final holdout metrics are computed once after
selection.
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
DECISIONS_OUT = DATA_DIR / "v16_total_search_decisions.parquet"
RESULTS_OUT = DATA_DIR / "results_v16_total_search.json"
BACKTEST_OUT = DATA_DIR / "backtest_v16_total_search_results.json"
REPORT_OUT = DOCS_DIR / "v16_total_search_report.md"
FIG_OUT = FIG_DIR / "v16_total_search_equity.png"

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
MAX_FIT_ROWS = 90_000
PAIR_CAP_GRID = [0.04, 0.06, 0.08]

RAW_FEATURES = [
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
]

V12_FEATURES = ["p_short", "p_flat", "p_long", "pred_z", "raw_conf", "edge", "confidence", "score"]


@dataclass(frozen=True)
class Candidate:
    family: str
    signal: str
    score_col: str
    confidence_col: str | None
    mode: str
    quantile: float
    max_gross: float
    pair_cap: float
    rebalance_bars: int
    confidence_threshold: float
    regime_mode: str


def xgb_regressor(target_name: str) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.032,
        subsample=0.86,
        colsample_bytree=0.86,
        min_child_weight=8,
        reg_alpha=0.08,
        reg_lambda=2.2,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        random_state=SEED + abs(hash(target_name)) % 10_000,
        verbosity=0,
    )


def xgb_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.036,
        subsample=0.86,
        colsample_bytree=0.86,
        min_child_weight=8,
        reg_alpha=0.08,
        reg_lambda=2.2,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        verbosity=0,
    )


def load_input() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run V12 first")
    df = pd.read_parquet(INPUT).sort_values(["timestamp", "pair"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    required = {"timestamp", "pair", "next_1bar_ret", "vol_24", "label_action", *RAW_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "score" not in df.columns and {"pred_z", "confidence"}.issubset(df.columns):
        df["score"] = df["pred_z"].astype(float) * df["confidence"].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    subset = [*RAW_FEATURES, "next_1bar_ret", "vol_24", "label_action"]
    return df.dropna(subset=subset).reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["vol_24"].replace(0.0, np.nan).fillna(out["vol_24"].median())
    for horizon in [3, 6, 12]:
        col = f"fwd_{horizon}bar_ret"
        if col in out.columns:
            out[f"target_z_{horizon}"] = (out[col].astype(float) / vol).clip(-6.0, 6.0)
    if "target_z_6bar" in out.columns and "target_z_6" not in out.columns:
        out["target_z_6"] = out["target_z_6bar"].clip(-6.0, 6.0)
    return out.replace([np.inf, -np.inf], np.nan)


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = df[df["timestamp"] <= FIT_END].copy()
    select = df[(df["timestamp"] >= SELECT_START) & (df["timestamp"] <= SELECT_END)].copy()
    holdout = df[df["timestamp"] >= HOLDOUT_START].copy()
    if fit.empty or select.empty or holdout.empty:
        raise ValueError("Empty fit/select/holdout split")
    return fit, select, holdout


def train_signal_models(fit: pd.DataFrame, select: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fit_train = fit.tail(MAX_FIT_ROWS).copy()
    x_fit = fit_train[RAW_FEATURES].to_numpy(dtype=np.float32)
    produced: list[str] = []
    targets = [c for c in ["target_z_3", "target_z_6", "target_z_12"] if c in fit_train.columns]
    print(f"Training XGB signals on rows={len(fit_train)} targets={targets}", flush=True)

    select_out = select.copy()
    holdout_out = holdout.copy()
    for target in targets:
        valid = fit_train[target].notna()
        model = xgb_regressor(target)
        print(f"  fit regressor {target} valid={int(valid.sum())}", flush=True)
        model.fit(x_fit[valid.to_numpy()], fit_train.loc[valid, target].to_numpy(dtype=np.float32))
        score_name = f"xgb_{target}"
        select_out[score_name] = model.predict(select_out[RAW_FEATURES].to_numpy(dtype=np.float32)).astype(float)
        holdout_out[score_name] = model.predict(holdout_out[RAW_FEATURES].to_numpy(dtype=np.float32)).astype(float)
        produced.append(score_name)

    print("  fit classifier action", flush=True)
    clf = xgb_classifier()
    clf.fit(x_fit, fit_train["label_action"].astype(int).to_numpy())
    for part in [select_out, holdout_out]:
        proba = clf.predict_proba(part[RAW_FEATURES].to_numpy(dtype=np.float32))
        if proba.shape[1] != 3:
            fixed = np.zeros((len(part), 3), dtype=float)
            for idx, cls in enumerate(clf.classes_):
                fixed[:, int(cls)] = proba[:, idx]
            proba = fixed
        part["xgb_cls_edge"] = proba[:, 2] - proba[:, 0]
        part["xgb_cls_conf"] = np.maximum(proba[:, 2], proba[:, 0]) * (1.0 - proba[:, 1])
    produced.append("xgb_cls_edge")

    for part in [select_out, holdout_out]:
        part["rule_momentum"] = part["mom_z_24"].astype(float)
        part["rule_reversal"] = -part["mom_z_24"].astype(float)
        part["rule_cross_strength"] = part["ret_z_cs"].astype(float)
        part["rule_breakout"] = (part["price_pos_72"].astype(float) - 0.5) * 2.0
        part["rule_mom_cross"] = 0.65 * part["mom_z_24"] + 0.35 * part["ret_z_cs"]
        if {"xgb_target_z_6", "xgb_cls_edge"}.issubset(part.columns):
            part["ens_raw_xgb"] = 0.65 * part["xgb_target_z_6"] + 0.35 * part["xgb_cls_edge"]
        if {"xgb_target_z_6", "rule_mom_cross"}.issubset(part.columns):
            part["ens_raw_rule"] = 0.60 * part["xgb_target_z_6"] + 0.40 * part["rule_mom_cross"]
        if {"pred_z", "xgb_target_z_6", "rule_mom_cross"}.issubset(part.columns):
            part["ens_aug_v12"] = 0.45 * part["pred_z"] + 0.35 * part["xgb_target_z_6"] + 0.20 * part["rule_mom_cross"]
    for col in ["rule_momentum", "rule_reversal", "rule_cross_strength", "rule_breakout", "rule_mom_cross", "ens_raw_xgb", "ens_raw_rule", "ens_aug_v12"]:
        if col in select_out.columns:
            produced.append(col)
    return select_out, holdout_out, sorted(set(produced))


def apply_regime_scale(work: pd.DataFrame, raw_position: pd.Series, mode: str, max_gross: float) -> pd.Series:
    if mode == "none":
        return raw_position
    market_vol = work.groupby("timestamp")["market_vol_24"].transform("mean")
    vol_hi = float(work["market_vol_24"].quantile(0.78))
    mret = work.groupby("timestamp")["market_ret_6"].transform("mean")
    btc = work.groupby("timestamp")["btc_ret_24"].transform("mean")
    scale = pd.Series(1.0, index=work.index)
    if mode == "reduce_chop":
        chop = (market_vol >= vol_hi) & (mret.abs() < work["market_ret_6"].abs().quantile(0.55))
        scale.loc[chop] = 0.35
    elif mode == "trend_bias":
        bull = (mret > 0) | (btc > 0)
        bear = (mret < 0) | (btc < 0)
        scale.loc[bull & raw_position.lt(0)] = 0.50
        scale.loc[bear & raw_position.gt(0)] = 0.50
    elif mode == "defensive":
        scale.loc[market_vol >= vol_hi] = 0.55
    else:
        raise ValueError(f"Unknown regime mode {mode}")
    return raw_position * scale.clip(0.0, max_gross)


def build_positions(df: pd.DataFrame, candidate: Candidate) -> pd.Series:
    timestamps = pd.Index(sorted(df["timestamp"].unique()), name="timestamp")
    ts_rank = {ts: idx for idx, ts in enumerate(timestamps)}
    work = df.copy()
    work["tidx"] = work["timestamp"].map(ts_rank).astype(int)
    reb = work[work["tidx"].mod(candidate.rebalance_bars).eq(0)].copy()
    if candidate.confidence_col is not None and candidate.confidence_col in reb.columns:
        reb = reb[reb[candidate.confidence_col].fillna(0.0) >= candidate.confidence_threshold].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)
    reb["rank_pct"] = reb.groupby("timestamp")[candidate.score_col].rank(pct=True, method="first")
    reb["side"] = 0.0
    if candidate.mode in {"long_short", "long_only"}:
        reb.loc[reb["rank_pct"] >= 1.0 - candidate.quantile, "side"] = 1.0
    if candidate.mode in {"long_short", "short_only"}:
        reb.loc[reb["rank_pct"] <= candidate.quantile, "side"] = -1.0
    reb = reb[reb["side"].ne(0.0)].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)
    vol = reb["vol_24"].replace(0.0, np.nan)
    reb["raw_weight"] = reb["side"] * reb[candidate.score_col].abs() / vol
    reb["raw_weight"] = reb["raw_weight"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = reb.groupby("timestamp")["raw_weight"].transform(lambda s: float(np.abs(s).sum()))
    reb["position"] = np.where(gross > 0.0, reb["raw_weight"] / gross * candidate.max_gross, 0.0)
    reb["position"] = reb["position"].clip(-candidate.pair_cap, candidate.pair_cap)

    wide = reb.pivot(index="timestamp", columns="pair", values="position").reindex(timestamps).ffill().fillna(0.0)
    long_exp = wide.clip(lower=0.0).sum(axis=1)
    short_exp = (-wide.clip(upper=0.0)).sum(axis=1)
    gross_exp = long_exp + short_exp
    scale = np.ones(len(wide), dtype=float)
    gross_arr = gross_exp.to_numpy(dtype=float)
    np.divide(candidate.max_gross, gross_arr, out=scale, where=gross_arr > candidate.max_gross)
    wide = wide.mul(scale, axis=0)
    flat = wide.stack().rename("position").reset_index()
    merged = df[["timestamp", "pair"]].reset_index().merge(flat, on=["timestamp", "pair"], how="left").sort_values("index")
    base = merged["position"].fillna(0.0).set_axis(df.index)
    adjusted = apply_regime_scale(work, base, candidate.regime_mode, candidate.max_gross)
    return adjusted.reindex(df.index).fillna(0.0)


def simulate(df: pd.DataFrame, position: pd.Series, strategy: str) -> dict[str, object]:
    work = df[["timestamp", "pair", "next_1bar_ret"]].copy()
    work["position"] = position.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    wide = work.pivot(index="timestamp", columns="pair", values="position").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    if len(turnover):
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
    metrics = {
        "strategy": strategy,
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
    return metrics


def selection_score(metrics: dict[str, object]) -> float:
    ret = float(metrics["total_return_pct"]) / 100.0
    sharpe = float(metrics["sharpe"])
    mdd = abs(float(metrics["max_drawdown_pct"]) / 100.0)
    turnover = float(metrics["avg_turnover"])
    return 2.2 * ret + 0.32 * sharpe - 1.1 * mdd - 3.5 * turnover


def candidate_grid(signal_cols: list[str], df: pd.DataFrame) -> list[Candidate]:
    candidates: list[Candidate] = []
    for score_col in signal_cols:
        if score_col not in df.columns:
            continue
        family = "augmented_v12" if score_col in {"pred_z", "edge", "score", "ens_aug_v12"} else "strict_raw"
        if score_col.startswith("xgb") or score_col.startswith("ens"):
            conf_options: list[tuple[str | None, list[float]]] = [("xgb_cls_conf", [0.00, 0.18, 0.30])]
        elif score_col in {"pred_z", "edge", "score"} and "confidence" in df.columns:
            conf_options = [("confidence", [0.20, 0.30, 0.40])]
        else:
            conf_options = [(None, [0.00])]
        for confidence_col, conf_values in conf_options:
            for mode in ["long_short", "long_only", "short_only"]:
                for quantile in [0.10, 0.15, 0.20]:
                    for max_gross in [0.20, 0.30, 0.50, 0.80]:
                        for pair_cap in PAIR_CAP_GRID:
                            if pair_cap * 15 < max_gross * 0.75:
                                continue
                            for rebalance in [6, 12, 18, 24]:
                                for regime_mode in ["none", "defensive", "reduce_chop", "trend_bias"]:
                                    for conf in conf_values:
                                        candidates.append(Candidate(family, score_col, score_col, confidence_col, mode, quantile, max_gross, pair_cap, rebalance, conf, regime_mode))
    return candidates


def search_candidates(select: pd.DataFrame, candidates: list[Candidate]) -> tuple[Candidate, Candidate, list[dict[str, object]]]:
    logs: list[dict[str, object]] = []
    best_raw: tuple[float, Candidate] | None = None
    best_any: tuple[float, Candidate] | None = None
    print(f"Searching candidates={len(candidates)} on select rows={len(select)}", flush=True)
    for idx, cand in enumerate(candidates, 1):
        pos = build_positions(select, cand)
        metrics = simulate(select, pos, cand.signal)
        score = selection_score(metrics)
        entry = {
            "rank_order": idx,
            "selection_score": round(score, 6),
            "family": cand.family,
            "signal": cand.signal,
            "mode": cand.mode,
            "quantile": cand.quantile,
            "max_gross": cand.max_gross,
            "pair_cap": cand.pair_cap,
            "rebalance_bars": cand.rebalance_bars,
            "confidence_col": cand.confidence_col,
            "confidence_threshold": cand.confidence_threshold,
            "regime_mode": cand.regime_mode,
            "select_return_pct": metrics["total_return_pct"],
            "select_sharpe": metrics["sharpe"],
            "select_mdd_pct": metrics["max_drawdown_pct"],
            "select_turnover": metrics["avg_turnover"],
        }
        logs.append(entry)
        if best_any is None or score > best_any[0]:
            best_any = (score, cand)
        if cand.family == "strict_raw" and (best_raw is None or score > best_raw[0]):
            best_raw = (score, cand)
        if idx % 250 == 0:
            print(f"  searched {idx}/{len(candidates)} best_any={best_any[0]:.4f}", flush=True)
    if best_raw is None or best_any is None:
        raise RuntimeError("No candidates searched")
    logs.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    return best_raw[1], best_any[1], logs[:80]


def cand_to_dict(cand: Candidate) -> dict[str, object]:
    return {
        "family": cand.family,
        "signal": cand.signal,
        "mode": cand.mode,
        "quantile": cand.quantile,
        "max_gross": cand.max_gross,
        "pair_cap": cand.pair_cap,
        "rebalance_bars": cand.rebalance_bars,
        "confidence_col": cand.confidence_col,
        "confidence_threshold": cand.confidence_threshold,
        "regime_mode": cand.regime_mode,
    }


def strip_curve(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_equity(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, metrics in results.items():
        curve = metrics.get("equity_curve", [])
        if curve:
            plt.plot(curve, label=name, linewidth=1.4)
    plt.title("V16 Total Search Holdout Equity Curves")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=160)
    plt.close()


def write_report(output: dict[str, object], ranked: list[dict[str, object]]) -> None:
    best = ranked[0]
    lines = [
        "# V16 总攻搜索实验报告",
        "",
        "## 1. 目标",
        "在 4h 高频数据上系统搜索信号模型、交易层、regime 控制和敞口参数，目标是在严格 2025+ holdout 上找出当前最快赚钱的候选方案。",
        "",
        "## 2. 时间切分",
        f"- Fit: <= {FIT_END}",
        f"- Select: {SELECT_START} 到 {SELECT_END}",
        f"- Holdout: >= {HOLDOUT_START}",
        "",
        "## 3. 方法边界",
        "V16 同时搜索 strict_raw 与 augmented_v12 两类候选。strict_raw 只用行情衍生特征；augmented_v12 允许使用 V12 冻结预测列，因此只能作为增强候选而非纯 raw 结论。所有最终指标只在 holdout 上计算一次。",
        "",
        "## 4. Holdout 排名",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 换手 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in ranked:
        lines.append(
            f"| {item['strategy']} | {item['total_return_pct']}% | {item['sharpe']} | {item['max_drawdown_pct']}% | {item['avg_gross_exposure']} | {item['avg_turnover']} |"
        )
    lines += [
        "",
        "## 5. 当前结论",
        f"本轮 holdout 第一名是 `{best['strategy']}`：收益 {best['total_return_pct']}%，Sharpe {best['sharpe']}，最大回撤 {best['max_drawdown_pct']}%。",
        "如果第一名属于 augmented_v12，需要继续用未来 paper trading 或更严格重训来确认；如果 strict_raw 也表现良好，说明纯行情特征路线更有研究价值。",
        "",
        "## 6. 输出文件",
        f"- `{RESULTS_OUT}`",
        f"- `{BACKTEST_OUT}`",
        f"- `{DECISIONS_OUT}`",
        f"- `{FIG_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[1/7] Load data", flush=True)
    df = add_targets(load_input())
    fit, select, holdout = split_frames(df)
    print(f"fit={len(fit)} select={len(select)} holdout={len(holdout)}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    print("[2/7] Train signal models", flush=True)
    select_pred, holdout_pred, produced = train_signal_models(fit, select, holdout)
    base_signals = [
        "rule_momentum", "rule_reversal", "rule_cross_strength", "rule_breakout", "rule_mom_cross",
        *produced,
    ]
    for col in ["pred_z", "edge", "score"]:
        if col in select_pred.columns:
            base_signals.append(col)
    signals = sorted(set([s for s in base_signals if s in select_pred.columns]))
    print(f"signals={signals}", flush=True)

    print("[3/7] Search select candidates", flush=True)
    candidates = candidate_grid(signals, select_pred)
    best_raw, best_any, top_logs = search_candidates(select_pred, candidates)
    print(f"best_raw={cand_to_dict(best_raw)}", flush=True)
    print(f"best_any={cand_to_dict(best_any)}", flush=True)

    print("[4/7] Apply selected candidates to holdout", flush=True)
    holdout_eval = holdout_pred.copy()
    pos_raw = build_positions(holdout_eval, best_raw)
    pos_any = build_positions(holdout_eval, best_any)
    holdout_eval["pos_V16_StrictRaw"] = pos_raw
    holdout_eval["pos_V16_BestAny"] = pos_any
    if "pos_V12_LowTurnover" in holdout_eval.columns:
        holdout_eval["pos_V12_LowTurnover_reference"] = holdout_eval["pos_V12_LowTurnover"]
    if "pos_V12_LowTurnover_Exploratory" in holdout_eval.columns:
        holdout_eval["pos_V12_LowTurnover30_reference"] = holdout_eval["pos_V12_LowTurnover_Exploratory"]
    holdout_eval.to_parquet(DECISIONS_OUT, index=False)

    print("[5/7] Simulate holdout", flush=True)
    results: dict[str, dict[str, object]] = {
        "V16-StrictRaw-SelectedOn2024": simulate(holdout_eval, pos_raw, "V16-StrictRaw-SelectedOn2024"),
        "V16-BestAny-SelectedOn2024": simulate(holdout_eval, pos_any, "V16-BestAny-SelectedOn2024"),
        "AlwaysFlat": simulate(holdout_eval, pd.Series(0.0, index=holdout_eval.index), "AlwaysFlat"),
    }
    if "pos_V12_LowTurnover" in holdout_eval.columns:
        results["V12-LT-reference"] = simulate(holdout_eval, holdout_eval["pos_V12_LowTurnover"], "V12-LT-reference")
    if "pos_V12_LowTurnover_Exploratory" in holdout_eval.columns:
        results["V12-LT30-reference"] = simulate(holdout_eval, holdout_eval["pos_V12_LowTurnover_Exploratory"], "V12-LT30-reference")

    ranked = sorted([strip_curve(v) for v in results.values()], key=lambda x: (float(x["total_return_pct"]), float(x["sharpe"])), reverse=True)
    for item in ranked:
        print(f"  {item['strategy']}: ret={item['total_return_pct']} sharpe={item['sharpe']} mdd={item['max_drawdown_pct']} gross={item['avg_gross_exposure']} turn={item['avg_turnover']}", flush=True)

    print("[6/7] Write outputs", flush=True)
    output = {
        "methodology": "V16 aggressive select-window search with strict 2025+ holdout; strict_raw and augmented_v12 families separated",
        "config": {
            "fit_period": f"{fit['timestamp'].min()} to {fit['timestamp'].max()}",
            "select_period": f"{select['timestamp'].min()} to {select['timestamp'].max()}",
            "holdout_period": f"{holdout['timestamp'].min()} to {holdout['timestamp'].max()}",
            "fit_rows": len(fit),
            "select_rows": len(select),
            "holdout_rows": len(holdout),
            "signals": signals,
            "best_raw_candidate": cand_to_dict(best_raw),
            "best_any_candidate": cand_to_dict(best_any),
            "top_select_candidates": top_logs,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "results": ranked,
    }
    RESULTS_OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    BACKTEST_OUT.write_text(json.dumps({k: strip_curve(v) for k, v in results.items()}, indent=2, ensure_ascii=False))
    plot_equity(results)
    write_report(output, ranked)
    print("[7/7] Done", flush=True)


if __name__ == "__main__":
    main()
