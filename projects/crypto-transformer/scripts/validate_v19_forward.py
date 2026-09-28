"""V19 Forward Validation.

Apply FROZEN V17-StrictRaw strategy parameters to genuinely new 4h bars
(after 2026-06-07 12:00, the end of all prior V12-V18 backtests).

NO parameter selection on forward data. NO model retraining. Just apply
pre-registered rules and report honestly.

Frozen candidate (from V17 best_raw, selected on 2024 select window):
    family:           strict_raw
    signal:           rule_reversal (= -mom_z_24)
    mode:             long_short
    quantile:         0.10
    max_gross:        0.15
    pair_cap:         0.06
    rebalance_bars:   36
    regime_mode:      reduce_chop

Also includes diagnostic variants:
    - V17-StrictRaw-Forward (canonical, rebalance=36)
    - V17-Reversal-DailyRebal (rebalance=1, to see signal every bar)
    - AlwaysFlat
    - EqualWeightLong20

Outputs:
    data/v19_forward_decisions.parquet
    data/results_v19_forward.json
    data/backtest_v19_forward_results.json
    docs/v19_forward_report.md
    docs/figures/v19_forward_equity.png
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure scripts/ and src/ on path
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import train_v12 as v12  # noqa: E402
import train_v17_speed_search as v17  # noqa: E402

# ── frozen forward cutoff ──
FORWARD_CUTOFF = pd.Timestamp("2026-06-07 12:00:00", tz="UTC")
FIT_END = pd.Timestamp("2024-07-01 00:00:00", tz="UTC")  # for regime thresholds

# ── frozen V17-StrictRaw candidate ──
FROZEN_V17 = v17.Candidate(
    family="strict_raw",
    signal="rule_reversal",
    score_col="rule_reversal",
    confidence_col=None,
    mode="long_short",
    quantile=0.10,
    max_gross=0.15,
    pair_cap=0.06,
    rebalance_bars=36,
    confidence_threshold=0.0,
    regime_mode="reduce_chop",
)

# diagnostic: every-bar rebalance to see signal on all forward bars
FROZEN_V17_DAILY = v17.Candidate(
    family="strict_raw",
    signal="rule_reversal",
    score_col="rule_reversal",
    confidence_col=None,
    mode="long_short",
    quantile=0.10,
    max_gross=0.15,
    pair_cap=0.06,
    rebalance_bars=1,
    confidence_threshold=0.0,
    regime_mode="reduce_chop",
)

PAIRS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX",
    "LINK", "DOT", "LTC", "UNI", "AAVE", "ATOM", "NEAR", "OP",
]

INITIAL_CAPITAL = 10_000.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FUNDING_PER_BAR = 0.0001 / 6.0
ANNUALIZATION = np.sqrt(365 * 6)


def build_full_feature_frame() -> pd.DataFrame:
    """Load extended raw 4h cache and compute V12 features, keeping tail bars.

    Unlike v12.build_feature_frame which drops the last ~12 bars per pair
    (missing forward labels), this version keeps ALL bars with valid features.
    Tail bars have NaN labels but valid features for inference.
    """
    import numpy as np
    raw = v12.load_4h_cache()
    frames = [v12.add_pair_features(pair, df) for pair, df in raw.items()]
    all_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp", "pair"])
        .reset_index(drop=True)
    )
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True)

    all_df["ret_rank_6"] = all_df.groupby("timestamp")["ret_6"].rank(pct=True)
    ret_mean = all_df.groupby("timestamp")["ret_6"].transform("mean")
    ret_std = all_df.groupby("timestamp")["ret_6"].transform("std").replace(0, np.nan)
    all_df["ret_z_cs"] = (all_df["ret_6"] - ret_mean) / ret_std
    all_df["vol_rank_24"] = all_df.groupby("timestamp")["vol_24"].rank(pct=True)

    btc = all_df[all_df["pair"] == "BTC/USDT"][
        ["timestamp", "ret_6", "ret_24", "vol_24"]
    ].copy()
    btc = btc.rename(columns={
        "ret_6": "btc_ret_6", "ret_24": "btc_ret_24", "vol_24": "btc_vol_24",
    })
    all_df = all_df.merge(btc, on="timestamp", how="left")
    all_df["market_ret_6"] = all_df.groupby("timestamp")["ret_6"].transform("mean")
    all_df["market_vol_24"] = all_df.groupby("timestamp")["log_ret_1"].transform("std")

    threshold = (v12.FEE_RATE + v12.SLIPPAGE_RATE) * 2.0 + v12.FUNDING_PER_BAR * v12.LABEL_HORIZON
    all_df["target_ret_6bar"] = all_df["fwd_6bar_ret"]
    all_df["target_z_6bar"] = all_df["target_ret_6bar"] / all_df["vol_24"].replace(0, np.nan)
    all_df["label_action"] = 1
    all_df.loc[all_df["target_ret_6bar"] > threshold, "label_action"] = 2
    all_df.loc[all_df["target_ret_6bar"] < -threshold, "label_action"] = 0

    keep = ["timestamp", "pair", "pair_id", "close",
            "next_1bar_ret", "fwd_3bar_ret", "fwd_6bar_ret", "fwd_12bar_ret",
            "target_ret_6bar", "target_z_6bar", "label_action"] + v12.FEATURE_COLS
    all_df = all_df[keep].replace([np.inf, -np.inf], np.nan)
    all_df = all_df.dropna(subset=v12.FEATURE_COLS).reset_index(drop=True)
    all_df["next_1bar_ret"] = all_df["next_1bar_ret"].fillna(0.0)
    all_df["vol_24"] = all_df["vol_24"].fillna(all_df["vol_24"].median())
    all_df["row_id"] = np.arange(len(all_df))
    all_df["rule_reversal"] = -all_df["mom_z_24"].astype(float)
    return all_df


def compute_regime_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    """Compute regime thresholds from fit period only (pre-registered)."""
    fit = frame[frame["timestamp"] <= FIT_END]
    return {
        "vol_hi": float(fit["market_vol_24"].quantile(0.75)),
        "mom_abs": float(fit["market_ret_6"].abs().quantile(0.58)),
        "btc_abs": float(fit["btc_ret_24"].abs().quantile(0.58)),
    }


def positions_equal_weight_long(frame: pd.DataFrame) -> pd.Series:
    """Equal weight long 20% gross across all 15 pairs."""
    n_pairs = frame["pair"].nunique()
    weight = 0.20 / n_pairs
    return pd.Series(weight, index=frame.index)


def positions_always_flat(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=frame.index)


def simulate_forward(
    frame: pd.DataFrame,
    position: pd.Series,
    strategy: str,
    *,
    risk_clip: bool = True,
) -> dict:
    """Simulate on forward window only (after FORWARD_CUTOFF)."""
    fwd = frame[frame["timestamp"] > FORWARD_CUTOFF].copy()
    fwd_pos = position.reindex(fwd.index).fillna(0.0)

    if risk_clip and "vol_24" in fwd.columns:
        eff_ret = v17.direction_clipped_return(
            fwd["next_1bar_ret"], fwd_pos, fwd["vol_24"]
        )
    else:
        eff_ret = fwd["next_1bar_ret"]

    gross_pnl = fwd_pos * eff_ret
    turnover = fwd_pos.groupby(fwd["timestamp"]).transform(lambda s: s.diff().abs().fillna(0.0).sum())
    cost = turnover * (FEE_RATE + SLIPPAGE_RATE)
    funding = fwd_pos.clip(upper=0.0).abs() * FUNDING_PER_BAR
    net = gross_pnl - cost - funding

    daily_net = net.groupby(fwd["timestamp"]).sum()
    equity = INITIAL_CAPITAL * (1.0 + daily_net).cumprod()
    total_ret = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0) if len(equity) > 0 else 0.0

    if daily_net.std() > 0:
        sharpe = float(daily_net.mean() / daily_net.std() * ANNUALIZATION)
    else:
        sharpe = 0.0

    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min()) if len(equity) > 0 else 0.0

    gross_exp = fwd_pos.abs().groupby(fwd["timestamp"]).sum().mean()
    long_exp = fwd_pos.clip(lower=0.0).groupby(fwd["timestamp"]).sum().mean()
    short_exp = (-fwd_pos.clip(upper=0.0)).groupby(fwd["timestamp"]).sum().mean()
    avg_turn = float(turnover.groupby(fwd["timestamp"]).sum().mean()) if len(turnover) > 0 else 0.0

    win = float((daily_net > 0).sum() / max(len(daily_net), 1))

    n_bars = len(daily_net)
    n_days = n_bars / 6.0 if n_bars > 0 else 0.0

    return {
        "strategy": strategy,
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "win_rate": round(win * 100, 1),
        "final_value": round(float(equity.iloc[-1]), 2) if len(equity) > 0 else INITIAL_CAPITAL,
        "avg_gross_exposure": round(float(gross_exp), 4),
        "avg_long": round(float(long_exp), 4),
        "avg_short": round(float(short_exp), 4),
        "avg_turnover": round(avg_turn, 4),
        "gross_pnl_sum": round(float(gross_pnl.sum()), 6),
        "cost_sum": round(float(cost.sum()), 6),
        "net_pnl_sum": round(float(net.sum()), 6),
        "n_forward_bars": n_bars,
        "n_forward_days": round(n_days, 2),
        "risk_clip": risk_clip,
    }


def main() -> None:
    print("=" * 70)
    print("V19 FORWARD VALIDATION")
    print("=" * 70)

    print("\n[1/5] Loading extended 4h cache and building features...")
    frame = build_full_feature_frame()
    print(f"  Feature frame: {len(frame)} rows, "
          f"{frame['timestamp'].nunique()} timestamps, "
          f"{frame['pair'].nunique()} pairs")
    print(f"  Period: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")

    forward = frame[frame["timestamp"] > FORWARD_CUTOFF]
    print(f"\n  Forward window: {len(forward)} rows, "
          f"{forward['timestamp'].nunique()} timestamps")
    print(f"  Forward period: {forward['timestamp'].min()} -> "
          f"{forward['timestamp'].max()}")
    n_fwd_bars = forward["timestamp"].nunique()
    n_fwd_days = n_fwd_bars / 6.0
    print(f"  Forward length: {n_fwd_bars} bars = {n_fwd_days:.1f} days")

    print("\n[2/5] Computing regime thresholds from fit period "
          f"(<= {FIT_END})...")
    thresholds = compute_regime_thresholds(frame)
    print(f"  thresholds: {thresholds}")

    print("\n[3/5] Computing frozen strategy positions on full timeline...")
    # V17 canonical (rebalance=36)
    pos_v17 = v17.build_positions(frame, FROZEN_V17)
    pos_v17 = v17.apply_regime_scale(
        frame, pos_v17, FROZEN_V17.regime_mode, FROZEN_V17.max_gross
    )
    print(f"  V17-StrictRaw (rebalance=36): "
          f"active={int((pos_v17 != 0).sum())} rows, "
          f"mean_abs={pos_v17.abs().mean():.4f}")

    # V17 diagnostic daily rebalance
    pos_v17d = v17.build_positions(frame, FROZEN_V17_DAILY)
    pos_v17d = v17.apply_regime_scale(
        frame, pos_v17d, FROZEN_V17_DAILY.regime_mode, FROZEN_V17_DAILY.max_gross
    )
    print(f"  V17-Reversal-DailyRebal (rebalance=1): "
          f"active={int((pos_v17d != 0).sum())} rows, "
          f"mean_abs={pos_v17d.abs().mean():.4f}")

    # Baselines
    pos_flat = positions_always_flat(frame)
    pos_ewl = positions_equal_weight_long(frame)

    print("\n[4/5] Simulating forward window "
          f"(> {FORWARD_CUTOFF})...")

    results = {}
    for name, pos, clip in [
        ("V17-StrictRaw-Forward", pos_v17, True),
        ("V17-Reversal-DailyRebal-Forward", pos_v17d, True),
        ("AlwaysFlat", pos_flat, True),
        ("EqualWeightLong20", pos_ewl, True),
    ]:
        r = simulate_forward(frame, pos, name, risk_clip=clip)
        results[name] = r
        print(f"\n  {name}:")
        print(f"    return={r['total_return_pct']:+.2f}%  "
              f"Sharpe={r['sharpe']:.2f}  MDD={r['max_drawdown_pct']:.2f}%")
        print(f"    gross_exp={r['avg_gross_exposure']:.4f}  "
              f"turnover={r['avg_turnover']:.4f}  "
              f"win={r['win_rate']:.1f}%")
        print(f"    net_pnl_sum={r['net_pnl_sum']:.6f}  "
              f"bars={r['n_forward_bars']}  days={r['n_forward_days']}")

    # Save decisions parquet
    fwd_frame = frame[frame["timestamp"] > FORWARD_CUTOFF].copy()
    fwd_frame["pos_V17_StrictRaw_Forward"] = (
        pos_v17.reindex(fwd_frame.index).fillna(0.0)
    )
    fwd_frame["pos_V17_DailyRebal_Forward"] = (
        pos_v17d.reindex(fwd_frame.index).fillna(0.0)
    )
    fwd_frame["pos_AlwaysFlat"] = 0.0
    fwd_frame["pos_EqualWeightLong20"] = (
        pos_ewl.reindex(fwd_frame.index).fillna(0.0)
    )

    decisions_path = PROJECT_ROOT / "data" / "v19_forward_decisions.parquet"
    fwd_frame.to_parquet(decisions_path, index=False)
    print(f"\n  Saved: {decisions_path} ({len(fwd_frame)} rows)")

    # Save results JSON
    results_path = PROJECT_ROOT / "data" / "results_v19_forward.json"
    with open(results_path, "w") as f:
        json.dump({
            "forward_cutoff": str(FORWARD_CUTOFF),
            "forward_bars": n_fwd_bars,
            "forward_days": round(n_fwd_days, 2),
            "cache_end": str(frame["timestamp"].max()),
            "regime_thresholds_frozen_from_fit": thresholds,
            "frozen_v17_candidate": {
                "family": FROZEN_V17.family,
                "signal": FROZEN_V17.signal,
                "mode": FROZEN_V17.mode,
                "quantile": FROZEN_V17.quantile,
                "max_gross": FROZEN_V17.max_gross,
                "pair_cap": FROZEN_V17.pair_cap,
                "rebalance_bars": FROZEN_V17.rebalance_bars,
                "regime_mode": FROZEN_V17.regime_mode,
            },
            "results": results,
            "caveat": (
                f"Only {n_fwd_bars} bars ({n_fwd_days:.1f} days) of forward data. "
                "Statistically insufficient for any conclusion. "
                "This is a pipeline test, not validation."
            ),
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    # Save backtest JSON
    bt_path = PROJECT_ROOT / "data" / "backtest_v19_forward_results.json"
    with open(bt_path, "w") as f:
        json.dump({"strategies": list(results.values())}, f, indent=2)

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pos in [
        ("V17-StrictRaw-Forward", pos_v17),
        ("V17-Reversal-DailyRebal-Forward", pos_v17d),
        ("EqualWeightLong20", pos_ewl),
    ]:
        fwd = frame[frame["timestamp"] > FORWARD_CUTOFF].copy()
        fp = pos.reindex(fwd.index).fillna(0.0)
        eff = v17.direction_clipped_return(fwd["next_1bar_ret"], fp, fwd["vol_24"])
        gross_pnl = fp * eff
        turn = fp.groupby(fwd["timestamp"]).transform(
            lambda s: s.diff().abs().fillna(0.0).sum()
        )
        cost = turn * (FEE_RATE + SLIPPAGE_RATE)
        funding = fp.clip(upper=0.0).abs() * FUNDING_PER_BAR
        net = gross_pnl - cost - funding
        daily = net.groupby(fwd["timestamp"]).sum()
        eq = INITIAL_CAPITAL * (1.0 + daily).cumprod()
        if len(eq) > 0:
            ax.plot(eq.index, eq.values, label=name, linewidth=1.5)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="AlwaysFlat")
    ax.set_title(f"V19 Forward Validation ({n_fwd_bars} bars, {n_fwd_days:.1f} days)")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = PROJECT_ROOT / "docs" / "figures" / "v19_forward_equity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Write report
    print("\n[5/5] Writing report...")
    report_path = PROJECT_ROOT / "docs" / "v19_forward_report.md"
    r = results

    report = f"""# V19 前瞻验证报告

## 目的

在所有历史回测（V12–V18）结束后，用**冻结参数**策略对 **2026-06-07 12:00 之后**的全新 4h 数据进行前瞻验证。不做任何参数选择或模型重训。

## 前瞻窗口

| 指标 | 值 |
|------|-----|
| 起始 | {FORWARD_CUTOFF} |
| 结束 | {frame['timestamp'].max()} |
| K 线数 | {n_fwd_bars} 根 (4h) |
| 天数 | {n_fwd_days:.1f} 天 |
| 15 币对行数 | {len(forward)} |

**⚠️ 统计学限制**：{n_fwd_bars} 根 K 线（{n_fwd_days:.1f} 天）远不足以验证任何策略。V17-StrictRaw 的 rebalance 周期为 36 根 K 线（6 天），在 {n_fwd_bars} 根 K 线窗口内最多触发 1 次调仓。本报告仅为管道测试，不代表统计意义上的验证结果。

## 冻结策略

### V17-StrictRaw-Forward

从 V17 搜索中在 2024 selection 窗口上选出的最佳 strict_raw 候选，参数完全冻结：

| 参数 | 值 |
|------|-----|
| 信号 | rule_reversal (= -mom_z_24) |
| 方向 | long_short |
| 分位 | top/bottom 10% |
| 最大敞口 | 15% |
| 单币上限 | 6% |
| 调仓周期 | 36 根 (6 天) |
| regime | reduce_chop |

Regime 阈值从 fit 期（≤ {FIT_END.date()}）冻结，不使用前瞻数据。

### V17-Reversal-DailyRebal-Forward

诊断变体：相同信号/参数但每根 K 线调仓（rebalance=1），用于观察信号在每根新 K 线上的表现。

## 结果

| 策略 | 收益 | Sharpe | MDD | 敞口 | 换手 | 胜率 |
|------|------|--------|-----|------|------|------|
| V17-StrictRaw-Forward | {r['V17-StrictRaw-Forward']['total_return_pct']:+.2f}% | {r['V17-StrictRaw-Forward']['sharpe']:.2f} | {r['V17-StrictRaw-Forward']['max_drawdown_pct']:.2f}% | {r['V17-StrictRaw-Forward']['avg_gross_exposure']:.4f} | {r['V17-StrictRaw-Forward']['avg_turnover']:.4f} | {r['V17-StrictRaw-Forward']['win_rate']:.1f}% |
| V17-Reversal-DailyRebal | {r['V17-Reversal-DailyRebal-Forward']['total_return_pct']:+.2f}% | {r['V17-Reversal-DailyRebal-Forward']['sharpe']:.2f} | {r['V17-Reversal-DailyRebal-Forward']['max_drawdown_pct']:.2f}% | {r['V17-Reversal-DailyRebal-Forward']['avg_gross_exposure']:.4f} | {r['V17-Reversal-DailyRebal-Forward']['avg_turnover']:.4f} | {r['V17-Reversal-DailyRebal-Forward']['win_rate']:.1f}% |
| EqualWeightLong20 | {r['EqualWeightLong20']['total_return_pct']:+.2f}% | {r['EqualWeightLong20']['sharpe']:.2f} | {r['EqualWeightLong20']['max_drawdown_pct']:.2f}% | {r['EqualWeightLong20']['avg_gross_exposure']:.4f} | {r['EqualWeightLong20']['avg_turnover']:.4f} | {r['EqualWeightLong20']['win_rate']:.1f}% |
| AlwaysFlat | 0.00% | 0.00 | 0.00% | 0.0000 | 0.0000 | — |

## 诚实结论

1. **数据量严重不足**：仅 {n_fwd_bars} 根 K 线（{n_fwd_days:.1f} 天），无法得出任何统计结论。
2. **Binance API 不可达**：当前环境无法从 Binance 拉取更多新数据，4h 缓存止于 {frame['timestamp'].max()}。
3. **V12-LT 无法前瞻**：V12 Transformer 模型 checkpoint 未保存，无法在新数据上生成预测。V12-LT/V12-LT30 的前瞻验证需要重训模型或部署 paper trading。
4. **管道已就绪**：本脚本可在新数据积累后重复运行，无需修改参数。
5. **真正验证需要**：至少 3–6 个月的前瞻数据（约 1080–2160 根 4h K 线），或部署 paper trading 系统实时积累。

## 输出文件

- `data/v19_forward_decisions.parquet`
- `data/results_v19_forward.json`
- `data/backtest_v19_forward_results.json`
- `docs/figures/v19_forward_equity.png`
"""

    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 70)
    print("V19 FORWARD VALIDATION COMPLETE")
    print(f"  Forward bars: {n_fwd_bars} ({n_fwd_days:.1f} days)")
    print(f"  V17-StrictRaw: {r['V17-StrictRaw-Forward']['total_return_pct']:+.2f}%")
    print("  ⚠️ Insufficient data for statistical conclusion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
