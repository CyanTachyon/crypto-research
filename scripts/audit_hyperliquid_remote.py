#!/usr/bin/env python3
"""Read-only audit of the Python project on U; NEVER import its trading modules.

Run from the local project:
  ssh U 'cd /home/cyan/default/crypto; .venv/bin/python -' \
    < scripts/audit_hyperliquid_remote.py > docs/hyperliquid-audit-evidence.json

Only selected function definitions are compiled from AST. Trading, clock, file
writes and notifications are replaced with local doubles. No .env is loaded.
The optional parquet analysis uses numpy/pandas from the remote virtualenv.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path.cwd()
evidence: dict = {"root": str(ROOT), "source_sha256": {}, "reproductions": []}


def extract(relative: str, names: set[str], namespace: dict) -> dict:
    source = (ROOT / relative).read_text()
    evidence["source_sha256"][relative] = hashlib.sha256(source.encode()).hexdigest()
    tree = ast.parse(source, filename=relative)
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in definitions} != names:
        raise ValueError(f"Missing audit functions in {relative}")
    # Future annotations prevent evaluation of annotations that refer to bot classes.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(module, relative, "exec"), namespace)
    return namespace


def record(name: str, reproduced: bool, **details) -> None:
    evidence["reproductions"].append({"issue": name, "reproduced": bool(reproduced), **details})


def trade(*args):
    return SimpleNamespace(net=args[7], fees=args[6], gross=args[5])


def new_state():
    return SimpleNamespace(balance=20.0, daily_pnl=0.0, daily_trades=0, position=None, trades=[])


class MockExchange:
    def __init__(self, partial=False):
        self.partial = partial
        self.protection_canceled = False
        self.remaining_size = 0.1

    def cancel_all_orders(self):
        self.protection_canceled = True
        return 2

    def limit_close(self, *_):
        status = {"error": "IOC did not fill"}
        if self.partial:
            self.remaining_size = 0.06
            status = {"filled": {"avgPx": "100", "totalSz": "0.04"}}
        return {"response": {"data": {"statuses": [status]}}}

    def get_position(self):
        return {"szi": str(self.remaining_size), "entryPx": "100"}

    def limit_open(self, *_):
        raise AssertionError("This audit must not attempt a live open")


def bot_reproductions():
    ns = extract("scripts/llm_scalper.py", {"sim_open", "sim_close"}, {
        "LIVE_TRADING": False, "LIVE": None,
        "MAX_DAILY_LOSS": 10.0, "MAX_MARGIN_USD": 10.0, "LEVERAGE": 1,
        "FEE_RATE": 0.0, "SLIPPAGE": 0.01, "COIN": "BTC",
        "time": SimpleNamespace(time=lambda: 1000.0, sleep=lambda _: None),
        "SimTrade": trade, "now_str": lambda: "AUDIT",
        "save_state": lambda _: None, "write_report": lambda _: None,
    })
    with contextlib.redirect_stdout(io.StringIO()):
        s = new_state()
        ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit")
        entry = s.position["entry"]
        ns["sim_close"](s, 100.0, "audit", "audit")
        record("favorable_slippage", s.balance > 20, entry=entry,
               equity_change_at_unchanged_market=s.balance - 20, expected="negative with positive slippage")

        ns.update(SLIPPAGE=0.0, FEE_RATE=0.001)
        s = new_state()
        ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit")
        ns["sim_close"](s, 100.0, "audit", "audit")
        record("entry_fee_missing_from_trade_net", not math.isclose(s.trades[0].net, s.balance - 20),
               reported_net=s.trades[0].net, equity_change=s.balance - 20)

        ns.update(FEE_RATE=0.0)
        s = new_state()
        ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit", limit_price=90.0)
        record("non_marketable_ioc_fills_in_sim", s.position is not None,
               current_price=100, buy_limit=90, simulated_fill=s.position["entry"])

        s = new_state()
        ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit")
        ns["sim_open"](s, "long", 100.0, 5.0, 10.0, 10.0, "audit")
        record("adding_overwrites_simulated_position", not math.isclose(s.position["size"], 0.15),
               actual_size=s.position["size"], expected_size=0.15)

        s = new_state()
        for _ in range(3):
            ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit")
        record("no_aggregate_margin_or_equity_check", s.position["margin_usd"] > s.balance,
               recorded_margin=s.position["margin_usd"], balance=s.balance)

        for partial in [False, True]:
            ns["LIVE_TRADING"] = False
            s = new_state()
            ns["sim_open"](s, "long", 100.0, 10.0, 10.0, 10.0, "audit")
            mock = MockExchange(partial=partial)
            ns.update(LIVE_TRADING=True, LIVE=mock)
            ns["sim_close"](s, 100.0, "audit", "audit")
            if partial:
                record("partial_close_discards_remaining_position", s.position is None and mock.remaining_size > 0,
                       exchange_remaining_size=mock.remaining_size, local_position=s.position)
            else:
                record("failed_close_leaves_unprotected_position", mock.protection_canceled and s.position is not None,
                       protection_canceled=mock.protection_canceled, exchange_remaining_size=mock.remaining_size)


def historical_diagnostics():
    import numpy as np
    import pandas as pd

    ns = extract("scripts/improve_v12_low_turnover.py", {"low_turnover_positions"}, {"np": np, "pd": pd})
    timestamps = pd.date_range("2025-01-01", periods=2, freq="4h")
    frame = pd.DataFrame([
        {"timestamp": timestamps[0], "pair": "BTC", "confidence": 1., "score": 1., "vol_24": .01},
        {"timestamp": timestamps[0], "pair": "ETH", "confidence": 1., "score": .1, "vol_24": .01},
        {"timestamp": timestamps[1], "pair": "BTC", "confidence": .1, "score": 1., "vol_24": .01},
        {"timestamp": timestamps[1], "pair": "ETH", "confidence": 1., "score": 1., "vol_24": .01},
    ])
    pos = ns["low_turnover_positions"](frame, confidence_threshold=.3, quantile=.2, max_gross=.2, rebalance_bars=1)
    record("deselected_assets_forward_filled_at_rebalance", float(pos.iloc[2]) > 0,
           btc_weight_before=float(pos.iloc[0]), btc_weight_after_confidence_fails=float(pos.iloc[2]))

    frame = pd.read_parquet(ROOT / "data/v14_holdout_decisions.parquet")
    strategies = ["pos_V12_LowTurnover", "pos_V12_LowTurnover_Exploratory", "pos_V14_V13_SelectedOn2024"]
    evidence["historical_scope"] = {
        "pairs": sorted(frame.pair.unique().tolist()),
        "start": str(frame.timestamp.min()), "end": str(frame.timestamp.max()),
        "weights_at_fixed_20_usdc_equity": {},
    }
    for col in strategies:
        weights = frame.loc[frame[col].abs() > 1e-12, col].abs()
        evidence["historical_scope"]["weights_at_fixed_20_usdc_equity"][col] = {
            "max_target_notional": float(weights.max() * 20),
            "fraction_target_positions_below_10": float((weights * 20 < 10).mean()),
        }
    ns = extract("scripts/train_v13.py", {"direction_clipped_return", "simulate"}, {
        "np": np, "pd": pd, "TAKE_VOL_MULT": 3., "STOP_VOL_MULT": 2.,
        "FEE_RATE": .001, "SLIPPAGE_RATE": .0005, "FUNDING_PER_BAR": .0001 / 6,
        "INITIAL_CAPITAL": 10000., "ANNUALIZATION": math.sqrt(365 * 6),
    })
    # Same frozen positions and old fee model; not a new executable backtest.
    comparisons = {}
    for clipped in [True, False]:
        result = ns["simulate"](frame, frame["pos_V14_V13_SelectedOn2024"], "diagnostic", risk_clip=clipped)
        comparisons[str(clipped)] = {k: result[k] for k in ["total_return_pct", "sharpe", "max_drawdown_pct"]}
    evidence["v14_return_clipping_diagnostic"] = comparisons


def saved_evidence():
    for name in ["paper_scalper_state.json", "llm_scalper_state.json"]:
        d = json.loads((ROOT / "data" / name).read_text())
        trades = d.get("trades", [])
        evidence[name] = {
            **{k: d[k] for k in ["balance", "daily_pnl", "total_trades", "decisions", "saved_at", "uptime_hours"] if k in d},
            "has_saved_position": bool(d.get("position")),
            "reported_net_sum": sum(t.get("net", t.get("net_pnl", 0)) for t in trades),
            "fees_sum": sum(t.get("fees", 0) for t in trades),
        }


if __name__ == "__main__":
    bot_reproductions()
    historical_diagnostics()
    saved_evidence()
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
