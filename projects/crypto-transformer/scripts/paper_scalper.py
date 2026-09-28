#!/usr/bin/env python3
"""Hyperliquid paper trading bot.

Uses REAL Hyperliquid prices but SIMULATES all trades internally.
No wallet, no private key, no real money. Pure logic verification.

Simulated costs (Hyperliquid mainnet):
  Taker fee:   0.035% per side (entry + exit)
  Slippage:    0.01% per side (conservative for small orders)

Run:  python scripts/paper_scalper.py
Stop: Ctrl+C
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOG_DIR = Path("data/paper_logs")
REPORT_FILE = Path("data/paper_scalper_report.md")
STATE_FILE = Path("data/paper_scalper_state.json")


def setup_logging():
    import logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"bot_{today}.log"
    logger = logging.getLogger("paper_scalper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    class StdoutRedirect:
        def write(self, text):
            for line in text.rstrip().splitlines():
                logger.info(line)
        def flush(self):
            pass

    sys.stdout = StdoutRedirect()
    sys.stderr = StdoutRedirect()
    return logger


LOG = None

COIN = "BTC"
LEVERAGE = 10
MARGIN_PER_TRADE = 15.0
POSITION_VALUE = MARGIN_PER_TRADE * LEVERAGE
TAKE_PROFIT_POINTS = 100.0
STOP_LOSS_POINTS = 500.0
ENTRY_PULLBACK_POINTS = 80.0
EMA_FAST = 30
EMA_SLOW = 120
TREND_TIMEFRAME = "1h"
TREND_REFRESH_SECONDS = 300
POLL_INTERVAL = 5
MAX_DAILY_LOSS = 5.0
MAX_DAILY_TRADES = 50
POSITION_TIMEOUT_MIN = 120
VOL_PAUSE_ATR_MULT = 2.5
FEE_RATE = 0.00035
SLIPPAGE = 0.0001
INITIAL_BALANCE = 100.0


@dataclass
class SimTrade:
    timestamp: str
    side: str
    entry: float
    exit_: float
    size: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    reason: str
    duration_s: float


@dataclass
class PaperState:
    balance: float = INITIAL_BALANCE
    price_history: list[float] = field(default_factory=list)
    candle_closes: list[float] = field(default_factory=list
)
    trend: str = "neutral"
    last_candle_refresh: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    daily_reset: str = ""
    sim_position: dict | None = None
    trades: list[SimTrade] = field(default_factory=list)
    running: bool = True
    loop_count: int = 0
    start_time: float = 0.0
    max_price_history: int = 360


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def compute_atr(prices: list[float], period: int = 20) -> float | None:
    if len(prices) < period + 1:
        return None
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    return float(np.mean(changes[-period:]))


def fetch_candles() -> list[float]:
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        end = int(time.time() * 1000)
        start = end - max(EMA_SLOW * 2, 240) * 3600 * 1000
        candles = info.candles_snapshot(COIN, TREND_TIMEFRAME, start, end)
        return [float(c["c"]) for c in candles]
    except Exception as e:
        print(f"[WARN] fetch_candles: {e}")
        return []


def fetch_price() -> float | None:
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        mids = info.all_mids()
        p = float(mids.get(COIN, 0))
        return p if p > 0 else None
    except Exception as e:
        print(f"[WARN] fetch_price: {e}")
        return None


def sim_open_short(state: PaperState, price: float) -> None:
    slip = price * (1 + SLIPPAGE)
    size = POSITION_VALUE / slip
    entry_fee = size * slip * FEE_RATE
    state.sim_position = {
        "side": "short",
        "entry": slip,
        "size": size,
        "entry_fee": entry_fee,
        "open_ts": time.time(),
    }
    state.balance -= entry_fee
    state.daily_trades += 1
    print(f"  [SIM OPEN SHORT] entry={slip:.1f} size={size:.5f} fee=${entry_fee:.4f}")


def sim_open_long(state: PaperState, price: float) -> None:
    slip = price * (1 - SLIPPAGE)
    size = POSITION_VALUE / slip
    entry_fee = size * slip * FEE_RATE
    state.sim_position = {
        "side": "long",
        "entry": slip,
        "size": size,
        "entry_fee": entry_fee,
        "open_ts": time.time(),
    }
    state.balance -= entry_fee
    state.daily_trades += 1
    print(f"  [SIM OPEN LONG] entry={slip:.1f} size={size:.5f} fee=${entry_fee:.4f}")


def sim_close(state: PaperState, exit_raw: float, reason: str) -> None:
    pos = state.sim_position
    if pos is None:
        return
    if pos["side"] == "short":
        slip_exit = exit_raw * (1 - SLIPPAGE)
        gross = (pos["entry"] - slip_exit) * pos["size"]
    else:
        slip_exit = exit_raw * (1 + SLIPPAGE)
        gross = (slip_exit - pos["entry"]) * pos["size"]
    exit_fee = pos["size"] * slip_exit * FEE_RATE
    net = gross - exit_fee
    state.balance += gross - exit_fee
    state.daily_pnl += net
    duration = time.time() - pos["open_ts"]
    trade = SimTrade(
        timestamp=now_str(), side=pos["side"], entry=pos["entry"],
        exit_=slip_exit, size=pos["size"], gross_pnl=gross,
        fees=pos["entry_fee"] + exit_fee,
        slippage_cost=abs(pos["entry"] - (pos["entry"] / (1 + SLIPPAGE) if pos["side"] == "short" else pos["entry"] / (1 - SLIPPAGE))) * pos["size"]
            + abs(slip_exit - exit_raw) * pos["size"],
        net_pnl=net, reason=reason, duration_s=duration,
    )
    state.trades.append(trade)
    state.sim_position = None
    save_state(state)
    write_report(state)
    print(f"  [SIM CLOSE {reason}] exit={slip_exit:.1f} "
          f"gross=${gross:+.4f} fees=${pos['entry_fee'] + exit_fee:.4f} "
          f"net=${net:+.4f} balance=${state.balance:.2f} "
          f"duration={duration:.0f}s")
    print(f"  [DAILY] trades={state.daily_trades} pnl=${state.daily_pnl:.2f}")


def manage_position(state: PaperState, price: float) -> bool:
    if state.sim_position is None:
        return False
    pos = state.sim_position
    entry = pos["entry"]
    side = pos["side"]
    elapsed_min = (time.time() - pos["open_ts"]) / 60.0

    if side == "short":
        tp = entry - TAKE_PROFIT_POINTS
        sl = entry + STOP_LOSS_POINTS
        if price <= tp:
            sim_close(state, price, "tp")
            return True
        if price >= sl:
            sim_close(state, price, "sl")
            return True
    else:
        tp = entry + TAKE_PROFIT_POINTS
        sl = entry - STOP_LOSS_POINTS
        if price >= tp:
            sim_close(state, price, "tp")
            return True
        if price <= sl:
            sim_close(state, price, "sl")
            return True

    if elapsed_min > POSITION_TIMEOUT_MIN:
        sim_close(state, price, "timeout")
        return True
    return False


def try_entry(state: PaperState, price: float) -> None:
    if state.trend == "bear" and len(state.price_history) >= 10:
        recent_high = max(state.price_history[-60:])
        if recent_high - price >= ENTRY_PULLBACK_POINTS:
            print(f"  [SIGNAL] bear regime, pulled back {recent_high - price:.0f}pts → SHORT")
            sim_open_short(state, price)

    elif state.trend == "bull" and len(state.price_history) >= 10:
        recent_low = min(state.price_history[-60:])
        if price - recent_low >= ENTRY_PULLBACK_POINTS:
            print(f"  [SIGNAL] bull regime, pulled back {price - recent_low:.0f}pts → LONG")
            sim_open_long(state, price)


def check_daily_reset(state: PaperState) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.daily_reset != today:
        state.daily_trades = 0
        state.daily_pnl = 0.0
        state.daily_reset = today
        print(f"\n[DAILY RESET] {today} | Balance: ${state.balance:.2f}")


def save_state(state: PaperState) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "balance": round(state.balance, 4),
        "daily_trades": state.daily_trades,
        "daily_pnl": round(state.daily_pnl, 4),
        "daily_reset": state.daily_reset,
        "trend": state.trend,
        "loop_count": state.loop_count,
        "start_time": state.start_time,
        "uptime_hours": round((time.time() - state.start_time) / 3600, 2) if state.start_time else 0,
        "total_trades": len(state.trades),
        "trades": [
            {
                "ts": t.timestamp, "side": t.side,
                "entry": round(t.entry, 2), "exit": round(t.exit_, 2),
                "net_pnl": round(t.net_pnl, 4), "fees": round(t.fees, 4),
                "reason": t.reason, "duration_s": round(t.duration_s, 1),
            }
            for t in state.trades
        ],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_report(state: PaperState) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades = state.trades
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    total_fees = sum(t.fees for t in trades)
    total_net = sum(t.net_pnl for t in trades)
    uptime_h = (time.time() - state.start_time) / 3600 if state.start_time else 0

    lines = [
        "# Hyperliquid 模拟交易报告",
        "",
        f"运行时间: {uptime_h:.1f} 小时",
        f"初始余额: ${INITIAL_BALANCE:.2f}",
        f"当前余额: ${state.balance:.2f}",
        f"总交易: {len(trades)} 笔",
        f"盈利: {len(wins)} 笔 | 亏损: {len(losses)} 笔 | 胜率: {len(wins)/max(len(trades),1)*100:.1f}%",
        f"总净盈亏: ${total_net:+.4f}",
        f"总手续费: ${total_fees:.4f}",
        f"手续费占比: {total_fees/max(abs(total_net)+total_fees, 0.01)*100:.1f}%",
        "",
        "## 最近 20 笔交易",
        "| 时间 | 方向 | 入场 | 出场 | 净盈亏 | 手续费 | 原因 | 时长 |",
        "|------|------|------|------|--------|--------|------|------|",
    ]
    for t in trades[-20:]:
        lines.append(
            f"| {t.timestamp} | {t.side} | {t.entry:.1f} | {t.exit_:.1f} | "
            f"${t.net_pnl:+.4f} | ${t.fees:.4f} | {t.reason} | {t.duration_s:.0f}s |"
        )
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global LOG
    LOG = setup_logging()
    LOG.info("=" * 60)
    LOG.info("  Hyperliquid PAPER TRADING Bot")
    LOG.info("  Real prices, simulated trades, zero risk")
    LOG.info("=" * 60)
    LOG.info(f"  Coin: {COIN} | Lev: {LEVERAGE}x | Margin: ${MARGIN_PER_TRADE}")
    LOG.info(f"  TP: {TAKE_PROFIT_POINTS}pts | SL: {STOP_LOSS_POINTS}pts")
    LOG.info(f"  Fee: {FEE_RATE*100}%/side | Slippage: {SLIPPAGE*100}%/side")
    LOG.info(f"  Trend: EMA{EMA_FAST}/{EMA_SLOW} on {TREND_TIMEFRAME}")
    LOG.info(f"  Initial balance: ${INITIAL_BALANCE}")
    LOG.info(f"  Poll: {POLL_INTERVAL}s | Trend refresh: {TREND_REFRESH_SECONDS}s")
    LOG.info(f"  Log file: {LOG_DIR}/bot_*.log")
    LOG.info(f"  State file: {STATE_FILE}")
    LOG.info(f"  Report file: {REPORT_FILE}")
    LOG.info("=" * 60)

    state = PaperState()
    state.start_time = time.time()

    def shutdown(sig, frame):
        print("\n[SHUTDOWN] Saving state and stopping...")
        state.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\nStarting paper trading. Press Ctrl+C to stop.\n")

    while state.running:
        try:
            state.loop_count += 1
            check_daily_reset(state)

            price = fetch_price()
            if price is None:
                time.sleep(POLL_INTERVAL * 2)
                continue

            state.price_history.append(price)
            if len(state.price_history) > state.max_price_history:
                state.price_history = state.price_history[-state.max_price_history:]

            if time.time() - state.last_candle_refresh > TREND_REFRESH_SECONDS:
                closes = fetch_candles()
                if len(closes) >= EMA_SLOW:
                    state.candle_closes = closes
                    ef = ema(closes, EMA_FAST)
                    es = ema(closes, EMA_SLOW)
                    if ef is not None and es is not None:
                        state.trend = "bear" if ef < es else ("bull" if ef > es else "neutral")
                    state.last_candle_refresh = time.time()
                    print(f"[{now_str()}] TREND={state.trend} "
                          f"EMA{EMA_FAST}={ef:.1f} EMA{EMA_SLOW}={es:.1f} "
                          f"price={price:.1f}")

            closed = manage_position(state, price)

            if state.sim_position is not None and not closed:
                pos = state.sim_position
                if pos["side"] == "short":
                    u = (pos["entry"] - price) * pos["size"]
                else:
                    u = (price - pos["entry"]) * pos["size"]
                if state.loop_count % 24 == 0:
                    print(f"  [MONITOR] price={price:.1f} pos={pos['side']} "
                          f"entry={pos['entry']:.1f} uPnL=${u:+.4f} "
                          f"open={(time.time()-pos['open_ts'])/60:.1f}min")
                time.sleep(POLL_INTERVAL)
                continue

            if state.daily_pnl <= -MAX_DAILY_LOSS:
                if state.loop_count % 60 == 0:
                    print(f"  [PAUSE] daily loss ${state.daily_pnl:.2f} <= -${MAX_DAILY_LOSS}")
                time.sleep(POLL_INTERVAL * 4)
                continue

            if state.daily_trades >= MAX_DAILY_TRADES:
                if state.loop_count % 60 == 0:
                    print(f"  [PAUSE] daily trades {state.daily_trades} >= {MAX_DAILY_TRADES}")
                time.sleep(POLL_INTERVAL * 4)
                continue

            if len(state.price_history) >= 60:
                short_atr = compute_atr(state.price_history[-20:], 20) or 0
                long_atr = compute_atr(state.price_history, 60) or 1
                if short_atr / max(long_atr, 1) > VOL_PAUSE_ATR_MULT:
                    if state.loop_count % 60 == 0:
                        print(f"  [VOL PAUSE] atr_ratio={short_atr/long_atr:.2f}")
                    time.sleep(POLL_INTERVAL * 2)
                    continue

            try_entry(state, price)

            if state.loop_count % 120 == 0:
                print(f"  [STATUS] price={price:.1f} trend={state.trend} "
                      f"trades={state.daily_trades} pnl=${state.daily_pnl:.2f} "
                      f"bal=${state.balance:.2f}")

            if state.loop_count % 360 == 0:
                save_state(state)
                write_report(state)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(10)

    save_state(state)
    write_report(state)
    print(f"\nPaper trading stopped.")
    print(f"  Total trades: {len(state.trades)}")
    print(f"  Final balance: ${state.balance:.2f}")
    print(f"  Report: {REPORT_FILE}")
    print(f"  State: {STATE_FILE}")


if __name__ == "__main__":
    main()
