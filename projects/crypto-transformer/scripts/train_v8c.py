#!/usr/bin/env python3
"""
V8-C: Statistical Arbitrage (Pairs Trading) for Crypto
=======================================================
Instead of predicting price direction, find co-integrated cryptocurrency pairs
and trade the mean-reversion of their price spread.

Approach: Engle-Granger + Johansen co-integration tests, z-score based signals,
walk-forward backtesting with transaction costs.
"""

import json
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, coint, OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = BASE_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT",
    "ADA_USDT", "AVAX_USDT", "LINK_USDT", "DOT_USDT", "UNI_USDT",
    "OP_USDT", "AAVE_USDT", "LTC_USDT", "ATOM_USDT", "NEAR_USDT",
]

TRAIN_CUTOFF = "2025-06-01"
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001  # 0.1% per trade

# Walk-forward parameters
FORMATION_PERIOD = 360   # days to estimate hedge ratio
TRADING_PERIOD = 60      # days to trade
STEP_FORWARD = 30        # roll forward by 30 days
ZSCORE_WINDOW = 20       # rolling window for z-score
ENTRY_THRESHOLD = 2.0    # z-score entry
EXIT_THRESHOLD = 0.5     # z-score exit
MAX_ACTIVE_PAIRS = 10    # cap on simultaneous pairs

# Co-integration filter criteria
ADF_PVALUE_THRESHOLD = 0.05
HALF_LIFE_MIN = 5
HALF_LIFE_MAX = 60
IN_SAMPLE_SHARPE_MIN = 0.5


# =============================================================================
# Step 1: Data Loading & Preprocessing
# =============================================================================
def load_data():
    """Load all 15 pairs' daily OHLCV, align timestamps, compute log prices."""
    print("=" * 70)
    print("V8-C: Statistical Arbitrage (Pairs Trading)")
    print("=" * 70)
    print("\n[Step 1] Loading data...")

    all_data = {}
    for pair in PAIRS:
        fpath = RAW_DIR / f"binance_{pair}_1d.parquet"
        df = pd.read_parquet(fpath)
        df = df[["timestamp", "close"]].copy()
        df.columns = ["timestamp", pair]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        all_data[pair] = df

    # Align timestamps via inner join
    price_df = pd.concat(all_data.values(), axis=1, join="inner")
    price_df = price_df.dropna()

    print(f"  Loaded {len(PAIRS)} pairs, {len(price_df)} common trading days")
    print(f"  Date range: {price_df.index.min().date()} to {price_df.index.max().date()}")

    # Compute log prices
    log_price_df = np.log(price_df)

    # Train/test split
    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz=price_df.index.tz)
    train_prices = price_df[price_df.index < cutoff]
    test_prices = price_df[price_df.index >= cutoff]
    train_log = log_price_df[log_price_df.index < cutoff]
    test_log = log_price_df[log_price_df.index >= cutoff]

    print(f"  Train: {train_prices.index.min().date()} to {train_prices.index.max().date()} ({len(train_prices)} days)")
    print(f"  Test:  {test_prices.index.min().date()} to {test_prices.index.max().date()} ({len(test_prices)} days)")

    return price_df, log_price_df, train_prices, test_prices, train_log, test_log


# =============================================================================
# Step 2: Co-integration Analysis
# =============================================================================
def compute_half_life(spread):
    """Compute half-life of mean reversion from AR(1) model."""
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()
    # Align
    common_idx = spread_lag.index.intersection(spread_ret.index)
    if len(common_idx) < 10:
        return np.inf
    spread_lag = spread_lag.loc[common_idx]
    spread_ret = spread_ret.loc[common_idx]

    # OLS: delta_spread = lambda * spread_lag + epsilon
    slope, _, _, _, _ = sp_stats.linregress(spread_lag.values, spread_ret.values)
    if slope >= 0:
        return np.inf  # not mean-reverting
    half_life = -np.log(2) / slope
    return half_life


def compute_in_sample_sharpe(spread, hedge_ratio, window=20):
    """Compute in-sample Sharpe from z-score strategy on spread."""
    if len(spread) < window + 10:
        return 0.0

    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std
    z_score = z_score.dropna()

    if len(z_score) < 10:
        return 0.0

    # Simple PnL: position = -sign(z) * min(|z|/2, 1)
    positions = -np.sign(z_score) * np.minimum(np.abs(z_score) / 2.0, 1.0)
    spread_returns = spread.diff().shift(-1).loc[z_score.index]
    pnl = positions * spread_returns
    pnl = pnl.dropna()

    if len(pnl) < 5 or pnl.std() < 1e-12:
        return 0.0

    sharpe = pnl.mean() / pnl.std() * np.sqrt(365)
    return sharpe


def engle_granger_test(log_a, log_b):
    """Run Engle-Granger co-integration test."""
    # Regression: log(A) = alpha + beta * log(B) + epsilon
    result = coint(log_a, log_b, trend='c', method='aeg')
    t_stat, p_value, crit_values = result
    return p_value, t_stat, crit_values


def johansen_test(log_a, log_b):
    """Run Johansen co-integration test (trace statistic)."""
    data = np.column_stack([log_a.values, log_b.values])
    # det_order=-1: no constant, 0: constant outside co-integration, 1: constant inside
    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=2)
        # Trace test for r=0 (no co-integration) and r<=1
        trace_stat_0 = result.lr1[0]   # test r=0
        trace_cv_0 = result.cvt[0]     # 90%, 95%, 99% critical values
        # Reject r=0 at 5% level?
        is_cointegrated_95 = trace_stat_0 > trace_cv_0[1]  # 95% CV
        return float(trace_stat_0), trace_cv_0.tolist(), is_cointegrated_95
    except Exception:
        return None, None, False


def run_cointegration_analysis(log_prices):
    """Run full co-integration analysis on ALL pair combinations."""
    print("\n[Step 2] Co-integration analysis (105 pairs)...")

    pairs_list = list(combinations(PAIRS, 2))
    results = []

    for i, (pair_a, pair_b) in enumerate(pairs_list):
        log_a = log_prices[pair_a].dropna()
        log_b = log_prices[pair_b].dropna()

        # Align
        common_idx = log_a.index.intersection(log_b.index)
        log_a = log_a.loc[common_idx]
        log_b = log_b.loc[common_idx]

        if len(log_a) < 100:
            continue

        # Engle-Granger
        eg_pvalue, eg_tstat, eg_crit = engle_granger_test(log_a, log_b)

        # OLS hedge ratio: log(A) = alpha + beta * log(B)
        slope, intercept, r_value, p_val_ols, std_err = sp_stats.linregress(log_b.values, log_a.values)
        hedge_ratio = slope

        # Compute spread and half-life
        spread = log_a - hedge_ratio * log_b
        half_life = compute_half_life(spread)

        # Johansen test
        joh_trace, joh_cv, joh_coint = johansen_test(log_a, log_b)

        # In-sample Sharpe
        in_sample_sharpe = compute_in_sample_sharpe(spread, hedge_ratio)

        result = {
            "pair": f"{pair_a}/{pair_b}",
            "pair_a": pair_a,
            "pair_b": pair_b,
            "eg_pvalue": float(eg_pvalue),
            "eg_tstat": float(eg_tstat),
            "hedge_ratio": float(hedge_ratio),
            "half_life": float(half_life) if np.isfinite(half_life) else None,
            "in_sample_sharpe": float(in_sample_sharpe),
            "johansen_trace_stat": joh_trace,
            "johansen_cointegrated_95": joh_coint,
            "r_squared": float(r_value ** 2),
            "n_obs": len(log_a),
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(pairs_list)} pairs...")

    # Sort by p-value
    results.sort(key=lambda x: x["eg_pvalue"])

    print(f"\n  Total pairs tested: {len(results)}")
    sig = [r for r in results if r["eg_pvalue"] < ADF_PVALUE_THRESHOLD]
    print(f"  Co-integrated at 5% level: {len(sig)}")

    # Apply all filters
    tradable = []
    for r in results:
        hl = r["half_life"]
        if hl is None:
            continue
        if (r["eg_pvalue"] < ADF_PVALUE_THRESHOLD
                and HALF_LIFE_MIN <= hl <= HALF_LIFE_MAX
                and r["in_sample_sharpe"] > IN_SAMPLE_SHARPE_MIN):
            tradable.append(r)

    print(f"  Tradable pairs (p<0.05, half-life 5-60d, Sharpe>0.5): {len(tradable)}")
    for t in tradable:
        print(f"    {t['pair']:25s}  p={t['eg_pvalue']:.4f}  beta={t['hedge_ratio']:.4f}  "
              f"HL={t['half_life']:.1f}d  Sharpe={t['in_sample_sharpe']:.2f}")

    return results, tradable


# =============================================================================
# Step 3 & 4: Walk-Forward Backtesting
# =============================================================================
def _screen_pairs_dynamic(formation_log, formation_price):
    """Screen ALL 105 pairs on formation period data, return co-integrated ones."""
    window_pairs = []
    for pair_a, pair_b in combinations(PAIRS, 2):
        log_a = formation_log[pair_a].dropna()
        log_b = formation_log[pair_b].dropna()
        common = log_a.index.intersection(log_b.index)

        if len(common) < 90:
            continue

        log_a = log_a.loc[common]
        log_b = log_b.loc[common]

        slope, intercept, r_val, _, _ = sp_stats.linregress(log_b.values, log_a.values)
        beta = slope
        spread = log_a - beta * log_b

        try:
            adf_result = adfuller(spread.values, maxlag=10, autolag='AIC')
            adf_pval = adf_result[1]
        except Exception:
            adf_pval = 1.0

        if adf_pval > 0.05:
            continue

        hl = compute_half_life(spread)
        if hl is None or hl > 120 or hl < 3:
            continue

        window_pairs.append({
            "pair_a": pair_a,
            "pair_b": pair_b,
            "pair_name": f"{pair_a}/{pair_b}",
            "hedge_ratio": beta,
            "half_life": hl,
            "adf_pval": adf_pval,
            "r_squared": r_val ** 2,
        })

    window_pairs.sort(key=lambda x: x["adf_pval"])
    return window_pairs[:MAX_ACTIVE_PAIRS]


def walk_forward_backtest(price_df, log_price_df, tradable_pairs):
    print("\n[Step 3/4] Walk-forward backtesting (dynamic pair selection)...")

    dates = price_df.index.sort_values()
    n_dates = len(dates)

    windows = []
    start_idx = 0
    while True:
        formation_end_idx = start_idx + FORMATION_PERIOD
        if formation_end_idx >= n_dates:
            break
        trading_start_idx = formation_end_idx
        trading_end_idx = min(trading_start_idx + TRADING_PERIOD, n_dates)

        windows.append({
            "formation_end_idx": formation_end_idx,
            "trading_start_idx": trading_start_idx,
            "trading_end_idx": trading_end_idx,
            "trading_dates": dates[trading_start_idx:trading_end_idx],
        })

        start_idx += STEP_FORWARD
        if trading_end_idx >= n_dates:
            break

    print(f"  Walk-forward windows: {len(windows)}")

    all_daily_pnl = {}
    all_positions = {}
    pair_hedge_hist = []
    pair_active_counts = {}
    all_pair_names_used = set()



    for wi, window in enumerate(windows):
        formation_idx = dates[:window["formation_end_idx"] + 1]
        trading_idx = window["trading_dates"]

        formation_log = log_price_df.loc[formation_idx]

        window_pairs = _screen_pairs_dynamic(formation_log, None)

        if (wi + 1) % 5 == 0 or wi == 0:
            print(f"    Window {wi+1}: {trading_idx[0].date()} to {trading_idx[-1].date()}, "
                  f"{len(window_pairs)} co-integrated pairs found")
            for wp in window_pairs[:3]:
                print(f"      {wp['pair_name']:25s}  p={wp['adf_pval']:.4f}  "
                      f"beta={wp['hedge_ratio']:.4f}  HL={wp['half_life']:.1f}d")

        for wp in window_pairs:
            all_pair_names_used.add(wp["pair_name"])
            pair_hedge_hist.append({
                "window": wi + 1,
                "pair": wp["pair_name"],
                "hedge_ratio": float(wp["hedge_ratio"]),
                "half_life": float(wp["half_life"]),
                "adf_pval": float(wp["adf_pval"]),
                "formation_end": str(formation_idx[-1].date()),
            })

        if not window_pairs:
            for dt in trading_idx:
                all_daily_pnl[dt] = {}
                all_positions[dt] = {}
                pair_active_counts[dt] = 0
            continue

        n_active = len(window_pairs)
        per_pair_weight = 1.0 / n_active

        current_positions = {}

        for dt in trading_idx:
            if dt not in all_daily_pnl:
                all_daily_pnl[dt] = {}
                all_positions[dt] = {}

            pair_active_counts[dt] = n_active
            dt_loc = price_df.index.get_loc(dt)

            for wp in window_pairs:
                pair_a, pair_b = wp["pair_a"], wp["pair_b"]
                pname = wp["pair_name"]
                beta = wp["hedge_ratio"]

                lookback_start = max(0, dt_loc - ZSCORE_WINDOW - 5)
                hist_log = log_price_df.iloc[lookback_start:dt_loc + 1]
                log_a_hist = hist_log[pair_a].dropna()
                log_b_hist = hist_log[pair_b].dropna()
                common = log_a_hist.index.intersection(log_b_hist.index)

                if len(common) < ZSCORE_WINDOW + 1:
                    continue

                log_a_hist = log_a_hist.loc[common]
                log_b_hist = log_b_hist.loc[common]

                spread_hist = log_a_hist - beta * log_b_hist
                rolling_mean = spread_hist.rolling(ZSCORE_WINDOW).mean()
                rolling_std = spread_hist.rolling(ZSCORE_WINDOW).std()

                current_spread = spread_hist.iloc[-1]
                current_mean = rolling_mean.iloc[-1]
                current_std = rolling_std.iloc[-1]

                if np.isnan(current_std) or current_std < 1e-10:
                    continue

                z = (current_spread - current_mean) / current_std

                if pname not in current_positions:
                    current_positions[pname] = 0

                pos = current_positions[pname]

                new_pos = pos
                if pos == 0:
                    if z > ENTRY_THRESHOLD:
                        new_pos = -1
                    elif z < -ENTRY_THRESHOLD:
                        new_pos = +1
                elif pos == +1:
                    if z > -EXIT_THRESHOLD:
                        new_pos = 0
                elif pos == -1:
                    if z < EXIT_THRESHOLD:
                        new_pos = 0

                prev_loc = dt_loc - 1
                if prev_loc < 0:
                    continue

                ret_a = price_df.iloc[dt_loc][pair_a] / price_df.iloc[prev_loc][pair_a] - 1
                ret_b = price_df.iloc[dt_loc][pair_b] / price_df.iloc[prev_loc][pair_b] - 1

                daily_pnl = 0.0
                if pos != 0:
                    spread_return = ret_a - beta * ret_b
                    daily_pnl = pos * spread_return * per_pair_weight

                if new_pos != pos:
                    daily_pnl -= abs(new_pos - pos) * FEE_RATE * per_pair_weight

                current_positions[pname] = new_pos
                all_daily_pnl[dt][pname] = daily_pnl
                all_positions[dt][pname] = new_pos

    sorted_dates = sorted(all_daily_pnl.keys())
    equity = [INITIAL_CAPITAL]
    daily_returns = []
    trade_count = 0

    for dt in sorted_dates:
        day_pnl = all_daily_pnl[dt]
        total_pnl = sum(day_pnl.values())
        daily_returns.append(total_pnl)
        equity.append(equity[-1] * (1 + total_pnl))

        for pname, pnl_val in day_pnl.items():
            if pnl_val != 0:
                trade_count += 1

    return {
        "equity": equity,
        "daily_returns": daily_returns,
        "dates": sorted_dates,
        "trade_count": trade_count,
        "pair_hedge_hist": pair_hedge_hist,
        "all_daily_pnl": all_daily_pnl,
        "all_positions": all_positions,
        "pair_active_counts": pair_active_counts,
        "n_tradable": len(all_pair_names_used),
    }


# =============================================================================
# Step 5: Buy & Hold Benchmark
# =============================================================================
def compute_benchmark(price_df, test_start):
    """Compute equal-weight buy & hold benchmark."""
    test_prices = price_df[price_df.index >= test_start]
    available_pairs = [p for p in PAIRS if p in test_prices.columns and test_prices[p].notna().any()]
    test_prices = test_prices[available_pairs].dropna()

    # Equal-weight daily returns
    daily_rets = test_prices.pct_change().dropna()
    avg_rets = daily_rets.mean(axis=1)

    equity = [INITIAL_CAPITAL]
    for r in avg_rets:
        equity.append(equity[-1] * (1 + r))

    return {
        "equity": equity,
        "daily_returns": avg_rets.tolist(),
        "dates": test_prices.index.tolist(),
    }


# =============================================================================
# Step 6: Per-Pair Performance Breakdown
# =============================================================================
def compute_per_pair_performance(backtest_results):
    """Break down performance by pair."""
    all_daily_pnl = backtest_results["all_daily_pnl"]
    pair_names = set()
    for dt, pnl_dict in all_daily_pnl.items():
        for k in pnl_dict:
            if k != "_n_active":
                pair_names.add(k)

    pair_perf = {}
    for pname in sorted(pair_names):
        daily_pnls = []
        for dt in sorted(all_daily_pnl.keys()):
            if pname in all_daily_pnl[dt]:
                daily_pnls.append(all_daily_pnl[dt][pname])

        if not daily_pnls:
            continue

        pnls = np.array(daily_pnls)
        cum_ret = np.prod(1 + pnls) - 1
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-12) * np.sqrt(365)
        win_rate = np.mean(pnls > 0) * 100
        n_trades = np.sum(pnls != 0)
        avg_holding_days = len(pnls) / max(n_trades, 1)

        pair_perf[pname] = {
            "total_return_pct": float(cum_ret * 100),
            "sharpe": float(sharpe),
            "win_rate_pct": float(win_rate),
            "n_trades": int(n_trades),
            "avg_holding_days": float(avg_holding_days),
            "pct_time_in_trade": float(n_trades / len(pnls) * 100) if len(pnls) > 0 else 0,
        }

    return pair_perf


# =============================================================================
# Step 6: Evaluation Metrics
# =============================================================================
def compute_metrics(equity, daily_returns, trade_count, name="Strategy"):
    """Compute standard evaluation metrics."""
    equity = np.array(equity)
    dr = np.array(daily_returns)

    total_return = (equity[-1] / equity[0] - 1) * 100

    # Annualized
    n_days = len(dr)
    if n_days > 1:
        annual_return = ((equity[-1] / equity[0]) ** (365 / n_days) - 1) * 100
    else:
        annual_return = 0.0

    sharpe = np.mean(dr) / (np.std(dr) + 1e-12) * np.sqrt(365)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100

    win_rate = np.mean(dr > 0) * 100 if len(dr) > 0 else 0

    # Average holding period
    n_trades = trade_count

    return {
        "strategy": name,
        "total_return_pct": round(float(total_return), 2),
        "annualized_return_pct": round(float(annual_return), 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "win_rate_pct": round(float(win_rate), 1),
        "n_trades": int(n_trades),
        "n_days": int(n_days),
        "final_value": round(float(equity[-1]), 2),
    }


# =============================================================================
# Step 7: Visualizations
# =============================================================================
def plot_cointegration_heatmap(coint_results, save_path):
    """Heatmap of ADF p-values for all pair combinations."""
    print("\n[Step 7] Generating visualizations...")

    # Build matrix
    pair_labels = [p.replace("_USDT", "") for p in PAIRS]
    n = len(PAIRS)
    pval_matrix = np.ones((n, n))

    for r in coint_results:
        ia = PAIRS.index(r["pair_a"])
        ib = PAIRS.index(r["pair_b"])
        pval_matrix[ia, ib] = r["eg_pvalue"]
        pval_matrix[ib, ia] = r["eg_pvalue"]

    np.fill_diagonal(pval_matrix, 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    # Log scale for better visualization
    log_pvals = -np.log10(pval_matrix + 1e-10)
    im = ax.imshow(log_pvals, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(pair_labels, fontsize=10)

    # Annotate
    for i in range(n):
        for j in range(n):
            if i != j:
                val = pval_matrix[i, j]
                color = 'white' if val < 0.01 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='-log10(p-value)')
    ax.set_title('Co-integration ADF p-values (Engle-Granger)\nGreen = co-integrated (low p-value)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_equity_curve(strategy_equity, strategy_dates, benchmark, save_path):
    """Portfolio equity curve vs Buy&Hold."""
    fig, ax = plt.subplots(figsize=(14, 7))

    strat_dates = [pd.Timestamp(d) for d in strategy_dates]
    ax.plot(strat_dates, strategy_equity[1:], label='V8-C Pairs Trading', linewidth=2, color='#2196F3')

    if benchmark["dates"]:
        bnh_dates = [pd.Timestamp(d) for d in benchmark["dates"]]
        # Align lengths
        bnh_equity = benchmark["equity"][:len(bnh_dates)+1]
        ax.plot(bnh_dates[:len(bnh_equity)-1], bnh_equity[1:],
                label='Buy & Hold (Equal-Weight)', linewidth=2, color='gray', linestyle='--')

    ax.set_title('V8-C Statistical Arbitrage vs Buy & Hold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spread_examples(log_price_df, pair_hedge_hist, backtest_results, save_path):
    if not pair_hedge_hist:
        print("  No pair hedge history to plot spreads")
        return

    pair_agg = {}
    for entry in pair_hedge_hist:
        pname = entry["pair"]
        if pname not in pair_agg:
            pair_agg[pname] = {"count": 0, "total_hl": 0, "latest_beta": 0, "latest_hl": 0}
        pair_agg[pname]["count"] += 1
        pair_agg[pname]["total_hl"] += entry["half_life"]
        pair_agg[pname]["latest_beta"] = entry["hedge_ratio"]
        pair_agg[pname]["latest_hl"] = entry["half_life"]

    sorted_pairs = sorted(pair_agg.items(), key=lambda x: x[1]["count"], reverse=True)
    selected = []
    for pname, info in sorted_pairs[:4]:
        parts = pname.split("/")
        selected.append({
            "pair_a": parts[0],
            "pair_b": parts[1],
            "pair_name": pname,
            "hedge_ratio": info["latest_beta"],
            "half_life": info["latest_hl"],
        })

    fig, axes = plt.subplots(len(selected), 2, figsize=(18, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for i, tp in enumerate(selected):
        pair_a, pair_b = tp["pair_a"], tp["pair_b"]
        pname = f"{pair_a}/{pair_b}"

        log_a = log_price_df[pair_a]
        log_b = log_price_df[pair_b]
        common = log_a.index.intersection(log_b.index)
        log_a = log_a.loc[common]
        log_b = log_b.loc[common]

        beta = tp["hedge_ratio"]
        spread = log_a - beta * log_b

        rolling_mean = spread.rolling(ZSCORE_WINDOW).mean()
        rolling_std = spread.rolling(ZSCORE_WINDOW).std()
        z_score = (spread - rolling_mean) / rolling_std

        # Left: spread with rolling mean
        ax = axes[i, 0]
        ax.plot(spread.index, spread.values, linewidth=1, color='#2196F3', label='Spread')
        ax.plot(rolling_mean.index, rolling_mean.values, linewidth=1, color='orange', linestyle='--', label='Rolling Mean')
        ax.axvline(x=pd.Timestamp(TRAIN_CUTOFF, tz=spread.index.tz), color='red', linestyle=':', alpha=0.7, label='Train/Test Split')
        ax.set_title(f'Spread: {pair_a.replace("_USDT","")} - {beta:.3f}×{pair_b.replace("_USDT","")}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: z-score with thresholds
        ax = axes[i, 1]
        z_clean = z_score.dropna()
        ax.plot(z_clean.index, z_clean.values, linewidth=0.8, color='#333333')
        ax.axhline(y=ENTRY_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'Entry ±{ENTRY_THRESHOLD}')
        ax.axhline(y=-ENTRY_THRESHOLD, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=EXIT_THRESHOLD, color='green', linestyle=':', alpha=0.7, label=f'Exit ±{EXIT_THRESHOLD}')
        ax.axhline(y=-EXIT_THRESHOLD, color='green', linestyle=':', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=pd.Timestamp(TRAIN_CUTOFF, tz=spread.index.tz), color='red', linestyle=':', alpha=0.7)
        ax.set_title(f'Z-Score (half-life={tp["half_life"]:.1f}d)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Example Co-integrated Pairs: Spread & Z-Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pair_performance(pair_perf, save_path):
    """Per-pair return bar chart."""
    if not pair_perf:
        print("  No pair performance data to plot")
        return

    names = [k.replace("_USDT", "").replace("/", " / ") for k in pair_perf.keys()]
    returns = [v["total_return_pct"] for v in pair_perf.values()]
    sharpes = [v["sharpe"] for v in pair_perf.values()]

    # Sort by return
    sorted_idx = np.argsort(returns)
    names = [names[i] for i in sorted_idx]
    returns = [returns[i] for i in sorted_idx]
    sharpes = [sharpes[i] for i in sorted_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.4)))

    colors = ['#4CAF50' if r > 0 else '#F44336' for r in returns]
    ax1.barh(names, returns, color=colors)
    ax1.set_xlabel('Total Return (%)')
    ax1.set_title('Per-Pair Total Return', fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    colors_s = ['#4CAF50' if s > 0 else '#F44336' for s in sharpes]
    ax2.barh(names, sharpes, color=colors_s)
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_title('Per-Pair Sharpe Ratio', fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle('V8-C Per-Pair Performance Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rolling_sharpe(daily_returns, dates, window=60, save_path=None):
    """Rolling Sharpe ratio over time."""
    dr = np.array(daily_returns)
    if len(dr) < window:
        print(f"  Not enough data for rolling Sharpe ({len(dr)} days, need {window})")
        return

    rolling_sharpe = []
    for i in range(window, len(dr) + 1):
        window_rets = dr[i - window:i]
        if np.std(window_rets) > 1e-12:
            rs = np.mean(window_rets) / np.std(window_rets) * np.sqrt(365)
        else:
            rs = 0.0
        rolling_sharpe.append(rs)

    rs_dates = [pd.Timestamp(d) for d in dates[window - 1:]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(rs_dates, rolling_sharpe, linewidth=1.5, color='#2196F3')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
    ax.axhline(y=-1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = -1.0')
    ax.fill_between(rs_dates, 0, rolling_sharpe, where=[r > 0 for r in rolling_sharpe],
                     alpha=0.2, color='green')
    ax.fill_between(rs_dates, 0, rolling_sharpe, where=[r <= 0 for r in rolling_sharpe],
                     alpha=0.2, color='red')

    ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio (Annualized)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    start_time = datetime.now()

    # Step 1: Load data
    price_df, log_price_df, train_prices, test_prices, train_log, test_log = load_data()

    # Step 2: Co-integration analysis on FULL dataset (for initial screening)
    coint_results, tradable_pairs = run_cointegration_analysis(log_price_df)

    # Save co-integration results
    coint_output = {
        "analysis_date": str(datetime.now()),
        "n_pairs_tested": len(coint_results),
        "n_cointegrated_5pct": sum(1 for r in coint_results if r["eg_pvalue"] < 0.05),
        "n_tradable": len(tradable_pairs),
        "filter_criteria": {
            "adf_pvalue_threshold": ADF_PVALUE_THRESHOLD,
            "half_life_range": [HALF_LIFE_MIN, HALF_LIFE_MAX],
            "min_in_sample_sharpe": IN_SAMPLE_SHARPE_MIN,
        },
        "all_results": coint_results,
        "tradable_pairs": tradable_pairs,
    }
    with open(DATA_DIR / "v8c_cointegration_results.json", "w") as f:
        json.dump(coint_output, f, indent=2, default=str)
    print(f"\n  Saved co-integration results to data/v8c_cointegration_results.json")

    # Step 3/4/5: Walk-forward backtesting
    backtest_results = walk_forward_backtest(price_df, log_price_df, tradable_pairs)

    # Step 5: Buy & Hold benchmark
    test_start = pd.Timestamp(TRAIN_CUTOFF, tz=price_df.index.tz)
    benchmark = compute_benchmark(price_df, test_start)

    # Step 6: Metrics
    strat_metrics = compute_metrics(
        backtest_results["equity"],
        backtest_results["daily_returns"],
        backtest_results["trade_count"],
        "V8-C Pairs Trading"
    )

    bnh_metrics = compute_metrics(
        benchmark["equity"],
        benchmark["daily_returns"],
        0,
        "Buy & Hold (Equal-Weight)"
    )

    per_pair = compute_per_pair_performance(backtest_results)

    all_positions = backtest_results["all_positions"]
    sorted_dates_pos = sorted(all_positions.keys())
    total_days_pos = len(sorted_dates_pos)
    days_in_trade = 0
    total_position_days = 0
    for dt in sorted_dates_pos:
        day_pos = all_positions[dt]
        active_positions = [v for v in day_pos.values() if v != 0]
        if active_positions:
            days_in_trade += 1
        total_position_days += len(active_positions)

    pct_time_in_trade = days_in_trade / max(total_days_pos, 1) * 100
    avg_positions_per_day = total_position_days / max(total_days_pos, 1)

    spread_stats = {
        "pct_time_in_trade": round(float(pct_time_in_trade), 1),
        "avg_positions_per_day": round(float(avg_positions_per_day), 2),
        "total_days": total_days_pos,
        "days_with_active_position": days_in_trade,
    }

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Strategy:     {strat_metrics['strategy']}")
    print(f"  Total Return: {strat_metrics['total_return_pct']:+.2f}%")
    print(f"  Annualized:   {strat_metrics['annualized_return_pct']:+.2f}%")
    print(f"  Sharpe:       {strat_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {strat_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:     {strat_metrics['win_rate_pct']:.1f}%")
    print(f"  Trades:       {strat_metrics['n_trades']}")
    print(f"  Final Value:  ${strat_metrics['final_value']:.2f}")
    print()
    print(f"  Benchmark:    {bnh_metrics['strategy']}")
    print(f"  Total Return: {bnh_metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe:       {bnh_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {bnh_metrics['max_drawdown_pct']:.2f}%")
    print()
    print(f"  V4 Reference: +14.04% return, 0.63 Sharpe, -29.81% MDD")
    print(f"  Active pairs: {backtest_results['n_tradable']}")

    if per_pair:
        print("\n  Per-Pair Performance:")
        for pname, perf in sorted(per_pair.items(), key=lambda x: x[1]["total_return_pct"], reverse=True):
            print(f"    {pname:25s}  ret={perf['total_return_pct']:+7.2f}%  "
                  f"sharpe={perf['sharpe']:5.2f}  win={perf['win_rate_pct']:4.1f}%  "
                  f"trades={perf['n_trades']:3d}")

    # Save results
    all_results = {
        "analysis_date": str(datetime.now()),
        "strategy": strat_metrics,
        "benchmark": bnh_metrics,
        "v4_reference": {
            "total_return_pct": 14.04,
            "sharpe": 0.63,
            "max_drawdown_pct": -29.81,
        },
        "per_pair_performance": per_pair,
        "spread_statistics": spread_stats,
        "walk_forward_config": {
            "formation_period": FORMATION_PERIOD,
            "trading_period": TRADING_PERIOD,
            "step_forward": STEP_FORWARD,
            "zscore_window": ZSCORE_WINDOW,
            "entry_threshold": ENTRY_THRESHOLD,
            "exit_threshold": EXIT_THRESHOLD,
            "max_active_pairs": MAX_ACTIVE_PAIRS,
        },
        "transaction_cost": FEE_RATE,
        "initial_capital": INITIAL_CAPITAL,
        "train_cutoff": TRAIN_CUTOFF,
        "test_period": f"{backtest_results['dates'][0].date()} to {backtest_results['dates'][-1].date()}" if backtest_results['dates'] else "N/A",
        "n_tradable_pairs": backtest_results["n_tradable"],
        "pair_hedge_history": backtest_results["pair_hedge_hist"],
        "average_holding_period_days": round(float(
            backtest_results["trade_count"] / max(total_days_pos, 1)
        ), 2),
    }

    with open(DATA_DIR / "results_v8c.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: data/results_v8c.json")

    # Save detailed backtest results
    detailed = {
        "equity_curve": [float(x) for x in backtest_results["equity"]],
        "daily_returns": [float(x) for x in backtest_results["daily_returns"]],
        "dates": [str(d) for d in backtest_results["dates"]],
    }
    with open(DATA_DIR / "backtest_v8c_results.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)
    print(f"  Saved: data/backtest_v8c_results.json")

    # Step 7: Visualizations
    plot_cointegration_heatmap(coint_results, FIG_DIR / "v8c_cointegration_heatmap.png")
    plot_equity_curve(
        backtest_results["equity"],
        backtest_results["dates"],
        benchmark,
        FIG_DIR / "v8c_equity_curve.png",
    )
    plot_spread_examples(
        log_price_df, backtest_results["pair_hedge_hist"], backtest_results,
        FIG_DIR / "v8c_spread_examples.png",
    )
    plot_pair_performance(per_pair, FIG_DIR / "v8c_pair_performance.png")
    plot_rolling_sharpe(
        backtest_results["daily_returns"],
        backtest_results["dates"],
        window=60,
        save_path=FIG_DIR / "v8c_rolling_sharpe.png",
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"V8-C complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    main()
