"""Causal hourly OHLC perpetual research simulator. No exchange execution code.

Signals passed to step MUST be from the preceding completed candle. Positions
carry signed price P&L, not spot purchase proceeds. Funding is settled before
boundary decisions using this hour's open as an explicitly approximate oracle.
No funding means an explicit zero rate, never an omitted record at integration.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional

COINS = ("BTC", "ETH")
HOUR = 3_600_000


@dataclass(frozen=True)
class Config:
    capital: float = 20.0
    max_notional: float = 15.0
    max_notional_ratio: float = 0.75
    risk_fraction: float = 0.01
    min_order: float = 10.0
    fee_bps: float = 4.5
    slippage_bps: float = 2.0
    max_drawdown: float = 0.10
    daily_loss_fraction: float = 0.03
    stop_atr: float = 3.0
    max_hold_hours: int = 168
    cooldown_hours: int = 4
    funding_cost_multiplier: float = 1.0
    funding_credit_fraction: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.capital <= 0 or self.max_notional <= 0 or self.min_order < 10:
            raise ValueError("Positive capital/notional and min_order >=10 required")
        if not 0 < self.max_notional_ratio <= 1 or not 0 < self.risk_fraction <= .02:
            raise ValueError("Research risk cap exceeded")
        if not 0 < self.max_drawdown <= .25 or not 0 < self.daily_loss_fraction <= .10:
            raise ValueError("Invalid account risk limits")
        if self.fee_bps > 100 or self.slippage_bps > 100 or self.stop_atr <= 0:
            raise ValueError("Invalid execution cost/stop")
        if self.funding_cost_multiplier < 1:
            raise ValueError("Funding multiplier must be conservative (>=1)")
        if self.funding_credit_fraction > 1:
            raise ValueError("Cannot amplify funding credits")


@dataclass
class Position:
    coin: str
    direction: int
    size: float
    entry: float
    stop: float
    entry_time: int
    entry_fee: float
    funding_paid: float = 0.0


def floor_size(size: float, decimals: int) -> float:
    if decimals not in range(7) or not math.isfinite(size) or size < 0:
        raise ValueError("Invalid size precision")
    return float(Decimal(str(size)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_DOWN))


class Engine:
    """Single position ledger. Each bar processed exactly once, sequentially."""

    def __init__(self, config: Config, sz_decimals: Dict[str, int]):
        if set(sz_decimals) != set(COINS) or any(v not in range(7) for v in sz_decimals.values()):
            raise ValueError("Only BTC and ETH metadata accepted")
        self.config = config
        self.sz_decimals = dict(sz_decimals)
        self.cash = config.capital
        self.peak = config.capital
        self.day_start_equity = config.capital
        self.day: Optional[int] = None
        self.day_blocked = False
        self.position: Optional[Position] = None
        self.last_time: Optional[int] = None
        self.next_entry_time = 0
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.halt_time: Optional[int] = None
        self.fees = 0.0
        self.funding_paid = 0.0
        self.realized_gross = 0.0
        self.max_dd = 0.0
        self.trades: list = []
        self.equity_curve: list = []
        self.skips: Dict[str, int] = {}
        self.last_prices: Dict[str, float] = {}

    def equity(self, prices: Dict[str, float]) -> float:
        p = self.position
        return self.cash + (p.direction * p.size * (prices[p.coin] - p.entry) if p else 0.)

    def _mark(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        self.peak = max(self.peak, eq)
        self.max_dd = max(self.max_dd, 1. - eq / self.peak)
        if self.day is not None and eq <= self.day_start_equity * (1. - self.config.daily_loss_fraction):
            self.day_blocked = True
        return eq

    def _skip(self, reason: str):
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def _fill(self, price: float, side: int) -> float:
        return price * (1. + side * self.config.slippage_bps / 10000.)

    def _close(self, price: float, time: int, reason: str):
        p = self.position
        if not p:
            return
        fill = self._fill(price, -p.direction)
        fee = p.size * fill * self.config.fee_bps / 10000.
        gross = p.direction * p.size * (fill - p.entry)
        self.cash += gross - fee
        self.realized_gross += gross
        self.fees += fee
        self.trades.append({
            **asdict(p), "exit": fill, "exit_time": time, "exit_reason": reason,
            "gross": gross, "fees": p.entry_fee + fee,
            "net": gross - p.entry_fee - fee - p.funding_paid,
        })
        self.position = None
        self.next_entry_time = time + self.config.cooldown_hours * HOUR
        self._mark(self.last_prices)

    def _halt(self, reason: str, time: int):
        self.halted = True
        self.halt_reason = reason
        if self.halt_time is None:
            self.halt_time = time

    def _try_open(self, coin: str, signal: Any, price: float, time: int):
        c = self.config
        if signal.direction not in (-1, 1) or not math.isfinite(signal.atr) or signal.atr <= 0:
            self._skip("invalid_signal")
            return
        fill = self._fill(price, signal.direction)
        distance = c.stop_atr * signal.atr
        stop = fill - signal.direction * distance
        if stop <= 0:
            self._skip("invalid_stop")
            return
        # Exact per-unit planned stop loss, including both fees and adverse
        # stop execution. Gap risk and future funding are not bounded by this.
        stop_fill = self._fill(stop, -signal.direction)
        stop_loss_per_unit = signal.direction * (fill - stop_fill) + (fill + stop_fill) * c.fee_bps / 10000.
        notional_cap = min(c.max_notional, c.max_notional_ratio * self.cash)
        raw_size = min(notional_cap / fill, self.cash * c.risk_fraction / stop_loss_per_unit)
        size = floor_size(max(0., raw_size), self.sz_decimals[coin])
        if size * fill < c.min_order:
            self._skip("below_min_notional")
            return
        fee = size * fill * c.fee_bps / 10000.
        if size * fill + fee > self.cash:
            self._skip("insufficient_unlevered_collateral")
            return
        self.cash -= fee
        self.fees += fee
        self.position = Position(coin, signal.direction, size, fill, stop, time, fee)

    def step(self, bars: Dict[str, dict], signals: Dict[str, Any], funding: Dict[str, float], decision=True,
             trailing_atrs: Optional[Dict[str, float]] = None):
        if set(bars) != set(COINS) or set(funding) != set(COINS):
            raise ValueError("Both BTC/ETH bars and explicit funding rates required")
        time = bars["BTC"]["t"]
        if not isinstance(time, int) or time % HOUR or bars["ETH"]["t"] != time:
            raise ValueError("Unaligned candle times")
        if self.last_time is not None and time != self.last_time + HOUR:
            raise ValueError("Non-sequential or duplicate bar")
        for coin in COINS:
            b = bars[coin]
            if any(not math.isfinite(b[k]) or b[k] <= 0 for k in ("o", "h", "l", "c")):
                raise ValueError("Invalid prices")
            if b["h"] < max(b["o"], b["c"], b["l"]) or b["l"] > min(b["o"], b["c"]):
                raise ValueError("Inconsistent OHLC")
            if not math.isfinite(funding[coin]):
                raise ValueError("Invalid funding")
        opens = {coin: bars[coin]["o"] for coin in COINS}
        closes = {coin: bars[coin]["c"] for coin in COINS}
        self.last_prices = opens
        opening_equity = self._mark(opens)
        day = time // (24 * HOUR)
        if day != self.day:
            self.day = day
            self.day_start_equity = opening_equity
            self.day_blocked = False

        # Boundary convention: fund the previously held position, then trade.
        p = self.position
        if p:
            cost = p.direction * p.size * opens[p.coin] * funding[p.coin]
            cost = cost * self.config.funding_cost_multiplier if cost > 0 else cost * self.config.funding_credit_fraction
            self.cash -= cost
            p.funding_paid += cost
            self.funding_paid += cost
        eq = self._mark(opens)
        closed = False
        if eq <= self.peak * (1. - self.config.max_drawdown):
            self._close(opens[p.coin], time, "account_drawdown_gap") if p else None
            self._halt("account_drawdown", time)
            closed = True

        p = self.position
        if p:
            signal = signals.get(p.coin)
            gap_hit = opens[p.coin] <= p.stop if p.direction == 1 else opens[p.coin] >= p.stop
            reverse = bool(signal and signal.direction == -p.direction)
            mean_exit = bool(signal and getattr(signal, "exit_long" if p.direction == 1 else "exit_short", False))
            timeout = time - p.entry_time >= self.config.max_hold_hours * HOUR
            if gap_hit or timeout or (decision and (reverse or mean_exit)):
                reason = "stop_gap" if gap_hit else "time_exit" if timeout else "signal_exit"
                self._close(opens[p.coin], time, reason)
                closed = True

        eq = self.equity(opens)
        day_blocked = self.day_blocked or eq <= self.day_start_equity * (1. - self.config.daily_loss_fraction)
        if not self.position and not closed and not self.halted and decision and time >= self.next_entry_time:
            if day_blocked:
                self._skip("daily_loss_limit")
            else:
                choices = [(s.strength, coin, s) for coin, s in signals.items()
                           if coin in COINS and s.direction in (-1, 1) and math.isfinite(s.strength)]
                if choices:
                    _, coin, signal = max(choices, key=lambda x: (x[0], -COINS.index(x[1])))
                    self._try_open(coin, signal, opens[coin], time)

        # Intrabar stop; no invented return clipping. Stop exits consume fees,
        # remove positions, and enforce cooldown. No take-profit ambiguity.
        p = self.position
        if p:
            b = bars[p.coin]
            floor_equity = self.peak * (1. - self.config.max_drawdown)
            account_stop = p.entry + (floor_equity - self.cash) / (p.direction * p.size)
            stop = max(p.stop, account_stop) if p.direction == 1 else min(p.stop, account_stop)
            account_tighter = stop != p.stop
            touched = b["l"] <= stop if p.direction == 1 else b["h"] >= stop
            if touched:
                # New entry can already lie through the account stop after fees.
                fill_ref = min(b["o"], stop) if p.direction == 1 else max(b["o"], stop)
                self._close(fill_ref, time + HOUR - 1, "account_drawdown" if account_tighter else "stop")
                if account_tighter:
                    self._halt("account_drawdown", time + HOUR - 1)
            else:
                adverse = {**opens, p.coin: b["l"] if p.direction == 1 else b["h"]}
                self._mark(adverse)

        self.last_prices = closes
        eq = self._mark(closes)
        if eq <= self.peak * (1. - self.config.max_drawdown):
            if self.position:
                self._close(closes[self.position.coin], time + HOUR - 1, "account_drawdown_close")
            self._halt("account_drawdown", time + HOUR - 1)
            eq = self.cash
        # Trailing stop uses the current completed close ONLY for next hour.
        if self.position:
            p = self.position
            signal = signals.get(p.coin)
            atr = trailing_atrs.get(p.coin, 0.) if trailing_atrs is not None else (signal.atr if signal else 0.)
            if math.isfinite(atr) and atr > 0:
                candidate = closes[p.coin] - p.direction * self.config.stop_atr * atr
                if candidate > 0:
                    p.stop = max(p.stop, candidate) if p.direction == 1 else min(p.stop, candidate)
        self.last_time = time
        self.equity_curve.append({"t": time + HOUR, "equity": eq,
                                  "gross_notional": self.position.size * closes[self.position.coin] if self.position else 0.})
        if not math.isclose(self.cash, self.config.capital + self.realized_gross - self.fees - self.funding_paid,
                            rel_tol=1e-10, abs_tol=1e-10):
            raise ArithmeticError("Ledger does not reconcile")

    def finish(self):
        if self.position:
            self._close(self.last_prices[self.position.coin], self.last_time + HOUR - 1, "end_of_sample")
            self.equity_curve[-1]["equity"] = self.cash
            self.equity_curve[-1]["gross_notional"] = 0.

    def report(self) -> dict:
        eq = self.equity(self.last_prices) if self.last_prices else self.cash
        positive = sum(max(0., t["net"]) for t in self.trades)
        negative = -sum(min(0., t["net"]) for t in self.trades)
        return {
            "config": asdict(self.config), "final_equity": eq,
            "return_pct": 100 * (eq / self.config.capital - 1.),
            "max_drawdown_pct": 100 * self.max_dd,
            "trade_count": len(self.trades),
            "fill_count": 2 * len(self.trades) + (1 if self.position else 0),
            "win_rate_pct": 100 * sum(t["net"] > 0 for t in self.trades) / len(self.trades) if self.trades else 0.,
            "profit_factor": positive / negative if negative > 0 else None,
            "fees": self.fees, "funding_paid": self.funding_paid,
            "realized_gross": self.realized_gross, "halted": self.halted, "halt_reason": self.halt_reason,
            "halt_time": self.halt_time,
            "skips": self.skips, "trades": self.trades, "equity_curve": self.equity_curve,
            "open_position": asdict(self.position) if self.position else None,
        }


def run_backtest(dataset: dict, signals: dict, config: Config, start: int, end: int,
                 decision_delay: int = 1) -> dict:
    """[start,end) hourly bars, signals i-delay; final positions liquidated."""
    assets = dataset["assets"]
    if decision_delay < 1 or not 1 <= start < end <= len(assets["BTC"]["candles"]):
        raise ValueError("Invalid causal test window")
    engine = Engine(config, {c: assets[c]["sz_decimals"] for c in COINS})
    funding = {}
    for c in COINS:
        funding[c] = {}
        for f in assets[c]["funding"]:
            t = f["t"] // HOUR * HOUR
            if t in funding[c]:
                raise ValueError("Duplicate hourly funding")
            funding[c][t] = f["rate"]
    for i in range(start, end):
        bars = {c: assets[c]["candles"][i] for c in COINS}
        t = bars["BTC"]["t"]
        if any(t not in funding[c] for c in COINS):
            raise ValueError("Missing actual hourly funding")
        sig = {c: signals[c][i - decision_delay] for c in COINS} if i >= decision_delay else {}
        engine.step(bars, sig, {c: funding[c][t] for c in COINS}, decision=t % (4 * HOUR) == 0,
                    trailing_atrs={c: signals[c][i].atr for c in COINS})
    engine.finish()
    return engine.report()
