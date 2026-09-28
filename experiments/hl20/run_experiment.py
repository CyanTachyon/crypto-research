"""Run the predeclared eight-candidate experiment. No credentials or order API.

Usage: python3 experiments/hl20/run_experiment.py --out experiments/hl20/results/v1
Creates a new output directory; refuses overwrites. Freezes manifests before
computing outcomes; no validation qualifier means no holdout trading evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from data_source import load_verified
from engine import COINS, HOUR, Config, floor_size, run_backtest
from strategies import CANDIDATES, build_signals

HERE = Path(__file__).resolve().parent
SCENARIOS = {
    "base": Config(),
    "moderate": Config(fee_bps=6, slippage_bps=5),
    "severe": Config(fee_bps=9, slippage_bps=10),
    "funding_adverse": Config(fee_bps=6, slippage_bps=5, funding_cost_multiplier=2, funding_credit_fraction=0),
    "delay_1h": Config(fee_bps=6, slippage_bps=5),
}


def iso(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brief(result):
    output = {k: v for k, v in result.items() if k not in ("trades", "equity_curve", "config")}
    trades, curve = result["trades"], result["equity_curve"]
    capital = result["config"]["capital"]
    daily_end = {}
    for p in curve:
        daily_end[(p["t"] - 1) // (24 * HOUR)] = p["equity"]
    previous = capital
    daily_returns = []
    for eq in daily_end.values():
        daily_returns.append(eq / previous - 1)
        previous = eq
    mean = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    variance = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns) if daily_returns else 0
    output.update({
        "net_usdc": result["final_equity"] - capital,
        "worst_daily_return_pct": 100 * min(daily_returns) if daily_returns else 0,
        "daily_sharpe_descriptive": mean / math.sqrt(variance) * math.sqrt(365) if variance > 0 else 0,
        "turnover_multiple": sum(t["size"] * (t["entry"] + t["exit"]) for t in trades) / capital,
        "mean_trade_net": sum(t["net"] for t in trades) / len(trades) if trades else 0,
        "end_of_hour_exposure_fraction": sum(p["gross_notional"] > 0 for p in curve) / len(curve) if curve else 0,
        "closed_trades_by_coin": {c: sum(t["coin"] == c for t in trades) for c in COINS},
        "net_by_coin": {c: sum(t["net"] for t in trades if t["coin"] == c) for c in COINS},
        "daily_returns": daily_returns,
    })
    return output


def passive(dataset, coin, start, end, config):
    """Separate 75%-initial-notional long-perp benchmark, held without stops.

    This is not a deployable strategy/risk recommendation. Both benchmarks
    start with 20 separately; they do not invest 30 from one 20 account.
    """
    asset = dataset["assets"][coin]
    rows = asset["candles"][start:end]
    rates = {f["t"] // HOUR * HOUR: f["rate"] for f in asset["funding"]}
    entry = rows[0]["o"] * (1 + config.slippage_bps / 10000)
    size = floor_size(min(config.max_notional, config.capital * config.max_notional_ratio) / entry,
                      asset["sz_decimals"])
    if size * entry < config.min_order:
        raise ValueError("Benchmark cannot satisfy min-order constraint")
    entry_fee = size * entry * config.fee_bps / 10000
    cash = config.capital - entry_fee
    paid, peak, maxdd = 0., config.capital, 0.
    curve = []
    for i, row in enumerate(rows):
        if i:
            funding = size * row["o"] * rates[row["t"]]
            funding = funding * config.funding_cost_multiplier if funding > 0 else funding * config.funding_credit_fraction
            cash -= funding
            paid += funding
        opened = cash + size * (row["o"] - entry)
        peak = max(peak, opened)
        low_eq = cash + size * (row["l"] - entry)
        maxdd = max(maxdd, 1 - low_eq / peak)
        eq = cash + size * (row["c"] - entry)
        peak = max(peak, eq)
        curve.append({"t": row["t"] + HOUR, "equity": eq, "gross_notional": size * row["c"]})
    exit_price = rows[-1]["c"] * (1 - config.slippage_bps / 10000)
    exit_fee = size * exit_price * config.fee_bps / 10000
    gross = size * (exit_price - entry)
    final = cash + gross - exit_fee
    maxdd = max(maxdd, 1 - final / peak)
    curve[-1].update(equity=final, gross_notional=0.)
    trade = {"coin": coin, "direction": 1, "size": size, "entry": entry, "exit": exit_price,
             "entry_time": rows[0]["t"], "exit_time": rows[-1]["t"] + HOUR - 1,
             "gross": gross, "fees": entry_fee + exit_fee,
             "funding_paid": paid, "net": final - config.capital, "exit_reason": "end_of_sample"}
    return {"config": asdict(config), "final_equity": final, "return_pct": 100 * (final / config.capital - 1),
            "max_drawdown_pct": 100 * maxdd, "trade_count": 1, "fill_count": 2, "win_rate_pct": 100 if final > config.capital else 0,
            "profit_factor": None, "fees": entry_fee + exit_fee, "funding_paid": paid,
            "realized_gross": gross, "halted": False, "skips": {}, "trades": [trade], "equity_curve": curve,
            "open_position": None, "benchmark_note": "75% initial long notional, no strategy stop/kill switch"}


def qualifier(base, moderate):
    reasons = []
    for label, r in (("base", base), ("moderate", moderate)):
        if r["return_pct"] <= 0:
            reasons.append(label + ":nonpositive_return")
        if r["max_drawdown_pct"] >= 8:
            reasons.append(label + ":drawdown_at_least_8pct")
        if r["trade_count"] < 10:
            reasons.append(label + ":fewer_than_10_trades")
        if r["halted"]:
            reasons.append(label + ":risk_halt")
    return reasons


def outcome_table(rows):
    output = ["| Candidate | Net USDC | Return | MDD | Trades | Fees | Funding paid |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in rows.items():
        output.append(f"| {name} | {r['net_usdc']:+.4f} | {r['return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | {r['trade_count']} | {r['fees']:.4f} | {r['funding_paid']:+.4f} |")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/hyperliquid_1h.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_verified(args.data)
    n = len(dataset["assets"]["BTC"]["candles"])
    if n < 2000:
        raise ValueError("Too little data for predeclared split")
    warmup = 336
    split1 = (warmup + (n - warmup) // 2) // 4 * 4
    split2 = (warmup + (n - warmup) * 3 // 4) // 4 * 4
    windows = {"development": (warmup, split1), "selection": (split1, split2), "holdout": (split2, n)}
    start_ms = dataset["range"]["start_ms"]
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "data_sha256": digest(args.data),
        "sources_sha256": {name: digest(HERE / name) for name in
                            ["PROTOCOL.md", "strategies.py", "engine.py", "run_experiment.py", "data_source.py",
                             "test_engine.py", "test_data_source.py", "test_strategies.py"]},
        "candidates": CANDIDATES, "scenarios": {k: asdict(v) for k, v in SCENARIOS.items()},
        "windows": {name: {"start_index": a, "end_index": b,
                           "start_utc": iso(start_ms + a * HOUR), "end_exclusive_utc": iso(start_ms + b * HOUR)}
                    for name, (a, b) in windows.items()},
        "decision_delay_hours": {name: 2 if name == "delay_1h" else 1 for name in SCENARIOS},
        "live_trading": False,
        "notes": ["Funding actual hourly rate uses trade-open proxy for missing historical oracle.",
                  "Funding belongs to previously held position before boundary entries; intra-hour timestamp offsets unresolved.",
                  "Hourly stops/market fills approximate executable prices; no L2 liquidity/partial-fill claim.",
                  "Retention provides about 208 days, not a complete multi-year market-cycle test.",
                  "Historical holdout is relative to this experiment only, never claimed as prospective.",
                  "No selection qualifier means no holdout strategy or benchmark returns are evaluated."],
    }
    write_json(args.out / "manifest.json", manifest)
    signals = build_signals(dataset)
    results = {"development": {}, "selection": {}, "holdout": {}, "benchmarks": {}}
    summaries = {"development": {}, "selection": {}, "holdout": {}, "benchmarks": {}}
    for phase in ("development", "selection"):
        a, b = windows[phase]
        for scenario in ("base", "moderate"):
            results[phase][scenario] = {}
            summaries[phase][scenario] = {}
            for candidate in CANDIDATES:
                result = run_backtest(dataset, signals[candidate], SCENARIOS[scenario], a, b)
                results[phase][scenario][candidate] = result
                summaries[phase][scenario][candidate] = brief(result)
        results["benchmarks"][phase], summaries["benchmarks"][phase] = {}, {}
        for coin in COINS:
            result = passive(dataset, coin, a, b, SCENARIOS["base"])
            results["benchmarks"][phase][coin] = result
            summaries["benchmarks"][phase][coin] = brief(result)
        print(f"Completed {phase}: {b-a} bars", flush=True)
    rejections, eligible = {}, []
    for candidate in CANDIDATES:
        base, mod = (summaries["selection"][scenario][candidate] for scenario in ("base", "moderate"))
        rejections[candidate] = qualifier(base, mod)
        if not rejections[candidate]:
            eligible.append(candidate)
    eligible.sort(key=lambda k: (-summaries["selection"]["moderate"][k]["return_pct"],
                                summaries["selection"]["moderate"][k]["max_drawdown_pct"],
                                summaries["selection"]["moderate"][k]["turnover_multiple"], CANDIDATES.index(k)))
    selected = eligible[0] if eligible else None
    selection = {"selected": selected, "eligible": eligible, "rejection_reasons": rejections,
                 "frozen_utc": datetime.now(timezone.utc).isoformat(), "holdout_evaluated": bool(selected)}
    # Persist selection before even computing the selected strategy's holdout.
    write_json(args.out / "selection.json", selection)
    gates = {"selection_passed": bool(selected), "historical_passed": False,
             "forward_paper_passed": False, "exchange_fault_tests_passed": False,
             "eligible_to_ask_for_real_trading": False}
    if selected:
        a, b = windows["holdout"]
        for scenario, config in SCENARIOS.items():
            result = run_backtest(dataset, signals[selected], config, a, b, 2 if scenario == "delay_1h" else 1)
            results["holdout"][scenario] = result
            summaries["holdout"][scenario] = brief(result)
        blocks = []
        for left in range(a, b, 15 * 24):
            right = min(b, left + 15 * 24)
            r = run_backtest(dataset, signals[selected], SCENARIOS["moderate"], left, right)
            blocks.append({"start_utc": iso(start_ms + left * HOUR), "end_utc": iso(start_ms + right * HOUR),
                           "full_15d": right - left == 15 * 24, **brief(r)})
        summaries["holdout_blocks"] = blocks
        full = [r for r in blocks if r["full_15d"]]
        gates.update({
            "all_holdout_stress_positive": all(r["return_pct"] > 0 for r in summaries["holdout"].values()),
            "holdout_base_drawdown_below_8pct": summaries["holdout"]["base"]["max_drawdown_pct"] < 8,
            "holdout_at_least_20_trades": summaries["holdout"]["base"]["trade_count"] >= 20,
            "majority_full_15d_blocks_positive": len(full) >= 2 and sum(r["return_pct"] > 0 for r in full) > len(full) / 2,
        })
        gates["historical_passed"] = all(gates[k] for k in (
            "all_holdout_stress_positive", "holdout_base_drawdown_below_8pct",
            "holdout_at_least_20_trades", "majority_full_15d_blocks_positive"))
        results["benchmarks"]["holdout"], summaries["benchmarks"]["holdout"] = {}, {}
        for coin in COINS:
            r = passive(dataset, coin, a, b, SCENARIOS["base"])
            results["benchmarks"]["holdout"][coin] = r
            summaries["benchmarks"]["holdout"][coin] = brief(r)
    summaries["selection_decision"], summaries["gates"] = selection, gates
    write_json(args.out / "results.json", results)
    write_json(args.out / "summary.json", summaries)
    lines = ["# Frozen BTC/ETH 20-USDC experiment", "", f"Selected: `{selected or 'NONE'}`", "",
             "Status: " + ("HISTORICAL_PASS_ONLY" if gates["historical_passed"] else "NOT_QUALIFIED"), "",
             "Real trading remains disabled. No real-trading approval requested.", "",
             "## Selection, base costs", "", outcome_table(summaries["selection"]["base"]), "",
             "## Selection, moderate cost stress", "", outcome_table(summaries["selection"]["moderate"]), "",
             "## Selection benchmarks (75% initial notional, no stop)", "",
             outcome_table(summaries["benchmarks"]["selection"]), "", "USDC cash benchmark: 0%.", ""]
    if selected:
        lines += ["## Selected candidate holdout", "", outcome_table(summaries["holdout"]), ""]
    else:
        lines += ["No candidate passed predeclared selection criteria. Holdout strategy returns remain unevaluated.", ""]
    lines += ["## Gate details", "", "```json", json.dumps(gates, indent=2), "```", "",
              "Full assumptions, windows and file hashes: manifest.json. Funding valuations use a trade-price proxy; hourly OHLC cannot validate exchange fills.", ""]
    (args.out / "RESULTS.md").write_text("\n".join(lines))
    write_json(args.out / "checksums.json", {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    print(json.dumps({"selected": selected, "gates": gates, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
