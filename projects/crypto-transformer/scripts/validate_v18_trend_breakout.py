"""V18 trend/breakout validation on 4h crypto data.

This script tests interpretable strategy families suggested after V16/V17:
Donchian breakout ensembles, EMA-confirmed breakout, time-series momentum,
and cross-sectional momentum. Parameter selection uses only the 2024 select
window, then applies the frozen winner once to the 2025+ holdout.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/v12_trade_decisions.parquet"
OUT_DECISIONS = ROOT / "data/v18_trend_breakout_decisions.parquet"
OUT_RESULTS = ROOT / "data/results_v18_trend_breakout.json"
OUT_BACKTEST = ROOT / "data/backtest_v18_trend_breakout_results.json"
OUT_REPORT = ROOT / "docs/v18_trend_breakout_report.md"
FIG_DIR = ROOT / "docs/figures"
FIG_OUT = FIG_DIR / "v18_trend_breakout_equity.png"

FIT_END = pd.Timestamp("2024-07-01 00:00:00")
SELECT_START = pd.Timestamp("2024-07-23 00:00:00")
SELECT_END = pd.Timestamp("2024-12-10 00:00:00")
HOLDOUT_START = pd.Timestamp("2025-01-01 00:00:00")

INITIAL_CAPITAL = 10_000.0
BAR_PER_DAY = 6
ANNUALIZATION = math.sqrt(365 * BAR_PER_DAY)
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_PER_BAR = 0.0001 / BAR_PER_DAY
STOP_VOL_MULT = 2.0
TAKE_VOL_MULT = 3.0


@dataclass(frozen=True)
class TrendCandidate:
    family: Literal["donchian", "ema_donchian", "tsmom", "cross_mom", "hybrid"]
    lookback: int
    slow: int
    quantile: float
    max_gross: float
    pair_cap: float
    rebalance_bars: int
    long_only: bool
    vol_window: int
    target_vol: float
    rebalance_threshold: float


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(INPUT).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    return df


def add_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    groups = work.groupby("pair", group_keys=False)
    # Donchian-style close breakout using close-only proxy because V12 table has no high/low.
    for lb in [30, 60, 120, 180, 360, 540]:
        roll_max = groups["close"].transform(lambda s: s.shift(1).rolling(lb, min_periods=max(10, lb // 3)).max())
        roll_min = groups["close"].transform(lambda s: s.shift(1).rolling(lb, min_periods=max(10, lb // 3)).min())
        sig = pd.Series(0.0, index=work.index)
        sig.loc[work["close"] > roll_max] = 1.0
        sig.loc[work["close"] < roll_min] = -1.0
        work[f"donchian_{lb}"] = sig
    for fast in [24, 48, 72]:
        fast_ema = groups["close"].transform(lambda s: s.ewm(span=fast, adjust=False, min_periods=fast).mean())
        for slow in [120, 180, 360]:
            slow_ema = groups["close"].transform(lambda s: s.ewm(span=slow, adjust=False, min_periods=slow).mean())
            work[f"ema_trend_{fast}_{slow}"] = np.sign(fast_ema - slow_ema).fillna(0.0)
    for lb in [42, 84, 126, 168, 252, 360]:
        mom = groups["close"].transform(lambda s: s / s.shift(lb) - 1.0)
        vol = groups["log_ret_1"].transform(lambda s: s.rolling(max(24, lb // 2), min_periods=max(12, lb // 4)).std())
        work[f"tsmom_{lb}"] = (mom / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Cross-sectional momentum features already exist, but normalize into stable scores.
    work["cross_mom_score"] = (0.6 * work["ret_z_cs"].fillna(0.0) + 0.4 * work["mom_z_24"].fillna(0.0))
    don_cols = [f"donchian_{lb}" for lb in [30, 60, 120, 180, 360]]
    work["donchian_ensemble"] = work[don_cols].mean(axis=1).fillna(0.0)
    work["hybrid_trend"] = 0.45 * work["donchian_ensemble"] + 0.35 * work["cross_mom_score"] + 0.20 * work["mom_z_24"].fillna(0.0)
    return work


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = df[df["timestamp"] <= FIT_END].copy()
    select = df[(df["timestamp"] >= SELECT_START) & (df["timestamp"] <= SELECT_END)].copy()
    holdout = df[df["timestamp"] >= HOLDOUT_START].copy()
    if fit.empty or select.empty or holdout.empty:
        raise RuntimeError("Invalid V18 split")
    return fit, select, holdout


def score_for_candidate(df: pd.DataFrame, cand: TrendCandidate) -> pd.Series:
    if cand.family == "donchian":
        return df[[f"donchian_{lb}" for lb in [30, 60, 120, cand.lookback] if f"donchian_{lb}" in df.columns]].mean(axis=1).fillna(0.0)
    if cand.family == "ema_donchian":
        trend_col = f"ema_trend_48_{cand.slow}"
        trend = df[trend_col] if trend_col in df.columns else pd.Series(0.0, index=df.index)
        base = df[f"donchian_{cand.lookback}"] if f"donchian_{cand.lookback}" in df.columns else df["donchian_ensemble"]
        return (base * (trend.abs() > 0).astype(float) * trend.replace(0.0, np.nan).fillna(base)).fillna(0.0)
    if cand.family == "tsmom":
        return df[f"tsmom_{cand.lookback}"].fillna(0.0)
    if cand.family == "cross_mom":
        return df["cross_mom_score"].fillna(0.0)
    if cand.family == "hybrid":
        return df["hybrid_trend"].fillna(0.0)
    raise ValueError(cand.family)


def build_positions(df: pd.DataFrame, cand: TrendCandidate) -> pd.Series:
    timestamps = pd.Index(sorted(df["timestamp"].unique()), name="timestamp")
    ts_rank = {ts: i for i, ts in enumerate(timestamps)}
    work = df.copy()
    work["score"] = score_for_candidate(work, cand)
    work["tidx"] = work["timestamp"].map(ts_rank).astype(int)
    reb = work[work["tidx"].mod(cand.rebalance_bars).eq(0)].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)
    reb["rank_pct"] = reb.groupby("timestamp")["score"].rank(pct=True, method="first")
    reb["side"] = 0.0
    if cand.long_only:
        reb.loc[(reb["rank_pct"] >= 1.0 - cand.quantile) & (reb["score"] > 0), "side"] = 1.0
    else:
        reb.loc[reb["rank_pct"] >= 1.0 - cand.quantile, "side"] = 1.0
        reb.loc[reb["rank_pct"] <= cand.quantile, "side"] = -1.0
    reb = reb[reb["side"].ne(0.0)].copy()
    if reb.empty:
        return pd.Series(0.0, index=df.index)
    vol = reb["vol_24"].replace(0.0, np.nan).fillna(reb["vol_24"].median())
    # Vol target at the asset level, then normalize by score and cap portfolio gross.
    ann_vol = (vol * ANNUALIZATION).clip(lower=0.05)
    vol_scale = (cand.target_vol / ann_vol).clip(upper=2.0)
    reb["raw_weight"] = reb["side"] * reb["score"].abs().clip(lower=0.05) * vol_scale
    gross = reb.groupby("timestamp")["raw_weight"].transform(lambda s: float(np.abs(s).sum()))
    reb["position"] = np.where(gross > 0.0, reb["raw_weight"] / gross * cand.max_gross, 0.0)
    reb["position"] = reb["position"].clip(-cand.pair_cap, cand.pair_cap)
    wide = reb.pivot(index="timestamp", columns="pair", values="position").reindex(timestamps).ffill().fillna(0.0)
    # Rebalance threshold: keep prior position if change is too small.
    if cand.rebalance_threshold > 0:
        prev = wide.shift(1).fillna(0.0)
        change = (wide - prev).abs()
        threshold = cand.rebalance_threshold * prev.abs().clip(lower=0.01)
        wide = wide.where(change >= threshold, prev)
    gross_exp = wide.abs().sum(axis=1)
    scale = np.ones(len(wide), dtype=float)
    np.divide(cand.max_gross, gross_exp.to_numpy(dtype=float), out=scale, where=gross_exp.to_numpy(dtype=float) > cand.max_gross)
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


def simulate(df: pd.DataFrame, position: pd.Series, strategy: str) -> dict[str, object]:
    work = df[["timestamp", "pair", "next_1bar_ret", "vol_24"]].copy()
    work["position"] = position.reindex(df.index).fillna(0.0).to_numpy(dtype=float)
    work["effective_ret"] = direction_clipped_return(work["next_1bar_ret"], work["position"], work["vol_24"])
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
    gross = float(metrics["avg_gross_exposure"])
    if sharpe < 0.5 or mdd > 0.12:
        return -999.0 + ret
    return 2.6 * ret + 0.22 * sharpe - 1.8 * mdd - 5.5 * turnover + 0.10 * gross


def candidate_grid() -> list[TrendCandidate]:
    out: list[TrendCandidate] = []
    for family in ["donchian", "ema_donchian", "tsmom", "cross_mom", "hybrid"]:
        lookbacks = [60, 120, 180, 360] if family in {"donchian", "ema_donchian"} else [84, 168, 252]
        if family in {"cross_mom", "hybrid"}:
            lookbacks = [120]
        for lb in lookbacks:
            for slow in [120, 180, 360]:
                if family != "ema_donchian" and slow != 120:
                    continue
                for quantile in [0.10, 0.20]:
                    for max_gross in [0.20, 0.30, 0.50]:
                        for pair_cap in [0.06, 0.10]:
                            if pair_cap * 15 < max_gross * 0.75:
                                continue
                            for rebalance in [12, 24, 36]:
                                for long_only in [False, True]:
                                    for vol_window in [540]:
                                        for target_vol in [0.20, 0.30]:
                                            for reb_th in [0.0, 0.20]:
                                                out.append(TrendCandidate(family, lb, slow, quantile, max_gross, pair_cap, rebalance, long_only, vol_window, target_vol, reb_th))
    return out


def search(select: pd.DataFrame, candidates: list[TrendCandidate]) -> tuple[TrendCandidate, list[dict[str, object]]]:
    best: tuple[float, TrendCandidate] | None = None
    logs: list[dict[str, object]] = []
    print(f"Searching V18 candidates={len(candidates)}", flush=True)
    for i, cand in enumerate(candidates, 1):
        pos = build_positions(select, cand)
        m = simulate(select, pos, str(cand.family))
        score = selection_score(m)
        entry = asdict(cand) | {
            "rank_order": i,
            "selection_score": round(score, 6),
            "select_return_pct": m["total_return_pct"],
            "select_sharpe": m["sharpe"],
            "select_mdd_pct": m["max_drawdown_pct"],
            "select_turnover": m["avg_turnover"],
            "select_avg_gross": m["avg_gross_exposure"],
        }
        logs.append(entry)
        if best is None or score > best[0]:
            best = (score, cand)
        if i % 1000 == 0:
            print(f"  searched {i}/{len(candidates)} best={best[0]:.4f}", flush=True)
    if best is None:
        raise RuntimeError("No V18 candidates")
    logs.sort(key=lambda x: float(x["selection_score"]), reverse=True)
    return best[1], logs[:100]


def strip_curve(metrics: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"daily_returns", "equity_curve"}}


def plot_equity(results: dict[str, dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for name, m in results.items():
        if m.get("equity_curve"):
            plt.plot(m["equity_curve"], label=name, linewidth=1.4)
    plt.title("V18 Trend Breakout Holdout Equity")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=160)
    plt.close()


def write_report(output: dict[str, object], ranked: list[dict[str, object]]) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V18 趋势/突破策略验证报告",
        "",
        "## 1. 方法",
        "V18 测试 Donchian、EMA+Donchian、时间序列动量、横截面动量和 hybrid 趋势策略。参数只在 2024 selection 窗口选择，2025+ holdout 一次性评估。",
        "",
        "## 2. 最佳 selection 候选",
        "```json",
        json.dumps(output["config"]["best_candidate"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. Holdout 排名",
        "| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 换手 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(f"| {row['strategy']} | {row['total_return_pct']}% | {row['sharpe']} | {row['max_drawdown_pct']}% | {row['avg_gross_exposure']} | {row['avg_turnover']} |")
    lines.extend([
        "",
        "## 4. 结论",
        "若 V18 未超过 V12-LT30-reference，则说明当前 4h 数据下最强候选仍是低换手 V12 冻结信号路线；趋势/突破可作为解释性基线而非最终冠军。",
    ])
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[1/6] Load and add signals", flush=True)
    df = add_trend_signals(load_frame())
    fit, select, holdout = split_frames(df)
    print(f"fit={len(fit)} select={len(select)} holdout={len(holdout)}", flush=True)
    print("[2/6] Search candidates on 2024 select", flush=True)
    best, logs = search(select, candidate_grid())
    print(f"best={asdict(best)}", flush=True)
    print("[3/6] Apply to holdout", flush=True)
    pos = build_positions(holdout, best)
    holdout_out = holdout.copy()
    holdout_out["pos_V18_TrendBreakout"] = pos.to_numpy(dtype=float)
    results = {
        "V18-TrendBreakout-SelectedOn2024": simulate(holdout_out, holdout_out["pos_V18_TrendBreakout"], "V18-TrendBreakout-SelectedOn2024"),
        "AlwaysFlat": simulate(holdout_out, pd.Series(0.0, index=holdout_out.index), "AlwaysFlat"),
    }
    if "pos_V12_LowTurnover" in holdout_out.columns:
        results["V12-LT-reference"] = simulate(holdout_out, holdout_out["pos_V12_LowTurnover"], "V12-LT-reference")
    if "pos_V12_LowTurnover_Exploratory" in holdout_out.columns:
        results["V12-LT30-reference"] = simulate(holdout_out, holdout_out["pos_V12_LowTurnover_Exploratory"], "V12-LT30-reference")
    ranked = sorted([strip_curve(v) for v in results.values()], key=lambda x: float(x["total_return_pct"]), reverse=True)
    print("[4/6] Results", flush=True)
    for r in ranked:
        print(f"  {r['strategy']}: ret={r['total_return_pct']} sharpe={r['sharpe']} mdd={r['max_drawdown_pct']} gross={r['avg_gross_exposure']} turn={r['avg_turnover']}", flush=True)
    print("[5/6] Write outputs", flush=True)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    holdout_out.to_parquet(OUT_DECISIONS, index=False)
    output = {
        "methodology": "V18 pre-registered trend/breakout family search on 2024 select, one-shot 2025+ holdout",
        "config": {
            "fit_period": f"{fit['timestamp'].min()} to {fit['timestamp'].max()}",
            "select_period": f"{select['timestamp'].min()} to {select['timestamp'].max()}",
            "holdout_period": f"{holdout['timestamp'].min()} to {holdout['timestamp'].max()}",
            "candidate_count": len(candidate_grid()),
            "best_candidate": asdict(best),
            "top_select_candidates": logs,
        },
        "holdout_ranked": ranked,
    }
    OUT_RESULTS.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_BACKTEST.write_text(json.dumps({k: strip_curve(v) for k, v in results.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_equity(results)
    write_report(output, ranked)
    print("[6/6] Done", flush=True)


if __name__ == "__main__":
    main()
