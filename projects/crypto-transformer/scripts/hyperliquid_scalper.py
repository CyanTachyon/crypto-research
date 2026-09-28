#!/usr/bin/env python3
"""Hyperliquid BTC scalping bot.

Automates the user's manual strategy:
- Detect short-term trend (EMA-based)
- In downtrend: short on bounces, TP 100 points, SL 500 points
- In uptrend: long on dips, TP 100 points, SL 500 points
- Hard risk limits: daily max loss, max trades, volatility filter

Usage:
    export HL_PRIVATE_KEY="your_wallet_private_key_here"
    export HL_CHAT_PRIVATE_KEY=""  # optional, for signing
    python scripts/hyperliquid_scalper.py

    # Or with a .env file containing the same vars.

SAFETY:
    - Reads private key ONLY from environment variable HL_PRIVATE_KEY
    - NEVER hardcodes keys in source code
    - Tests connection and prints position before starting
    - Has a KILL SWITCH: set MAX_DAILY_LOSS_USD appropriately
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


# ──────────────────────────────────────────────────────────────────────
# Configuration (edit these or use environment variables)
# ──────────────────────────────────────────────────────────────────────

COIN = os.environ.get("HL_COIN", "BTC")

LEVERAGE = int(os.environ.get("HL_LEVERAGE", "10"))
MARGIN_PER_TRADE = float(os.environ.get("HL_MARGIN_PER_TRADE", "15"))
TAKE_PROFIT_POINTS = float(os.environ.get("HL_TAKE_PROFIT_POINTS", "100"))
STOP_LOSS_POINTS = float(os.environ.get("HL_STOP_LOSS_POINTS", "500"))

EMA_FAST = int(os.environ.get("HL_EMA_FAST", "30"))
EMA_SLOW = int(os.environ.get("HL_EMA_SLOW", "120"))
ENTRY_PULLBACK_POINTS = float(os.environ.get("HL_ENTRY_PULLBACK_POINTS", "80"))

MAX_DAILY_LOSS_USD = float(os.environ.get("HL_MAX_DAILY_LOSS_USD", "5"))
MAX_DAILY_TRADES = int(os.environ.get("HL_MAX_DAILY_TRADES", "50"))
VOLATILITY_PAUSE_ATR_MULT = float(os.environ.get("HL_VOLATILITY_PAUSE_ATR_MULT", "2.5"))

POSITION_TIMEOUT_MINUTES = int(os.environ.get("HL_POSITION_TIMEOUT_MINUTES", "120"))
TREND_TIMEFRAME = os.environ.get("HL_TREND_TIMEFRAME", "1h")
TREND_REFRESH_SECONDS = int(os.environ.get("HL_TREND_REFRESH_SECONDS", "300"))
POLL_INTERVAL_SECONDS = int(os.environ.get("HL_POLL_INTERVAL_SECONDS", "5"))

LOG_DIR = Path(os.environ.get("HL_LOG_DIR", "data/scalper_logs"))


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    timestamp: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    exit_reason: str
    duration_seconds: float


@dataclass
class BotState:
    running: bool = True
    price_history: deque = field(default_factory=lambda: deque(maxlen=360))
    candle_closes: list[float] = field(default_factory=list)
    current_trend: str = "neutral"
    last_candle_refresh: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    daily_reset_ts: str = ""
    open_position: dict | None = None
    trade_history: list[TradeRecord] = field(default_factory=list)
    last_trade_ts: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Exchange wrapper
# ──────────────────────────────────────────────────────────────────────

class HyperliquidBot:
    def __init__(self) -> None:
        private_key = os.environ.get("HL_PRIVATE_KEY", "")
        if not private_key:
            print("ERROR: Set HL_PRIVATE_KEY environment variable first!")
            print("Example: export HL_PRIVATE_KEY='0xabc...'")
            sys.exit(1)

        try:
            from eth_account import Account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
        except ImportError:
            print("ERROR: Run: pip install hyperliquid-python-sdk")
            sys.exit(1)

        self.wallet = Account.from_key(private_key)
        self.address = self.wallet.address
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.wallet, constants.MAINNET_API_URL)

        print(f"Wallet address: {self.address}")
        print(f"Trading: {COIN} | Leverage: {LEVERAGE}x | Margin/trade: ${MARGIN_PER_TRADE}")

    def get_price(self) -> float | None:
        try:
            mids = self.info.all_mids()
            price = float(mids.get(COIN, 0))
            return price if price > 0 else None
        except Exception as e:
            print(f"[WARN] get_price failed: {e}")
            return None

    def get_candles(self, timeframe: str = "1h", count: int = 200) -> list[float]:
        try:
            import time as _time
            end_ms = int(_time.time() * 1000)
            start_ms = end_ms - count * 3600 * 1000 * 2
            candles = self.info.candles_snapshot(COIN, timeframe, start_ms, end_ms)
            closes = [float(c["c"]) for c in candles]
            return closes
        except Exception as e:
            print(f"[WARN] get_candles failed: {e}")
            return []

    def get_position(self) -> dict | None:
        try:
            user_state = self.info.user_state(self.address)
            positions = user_state.get("assetPositions", [])
            for pos in positions:
                p = pos.get("position", {})
                if p.get("coin") == COIN:
                    return p
            return None
        except Exception as e:
            print(f"[WARN] get_position failed: {e}")
            return None

    def set_leverage(self) -> bool:
        try:
            result = self.exchange.update_leverage(LEVERAGE, COIN)
            return True
        except Exception as e:
            print(f"[WARN] set_leverage failed: {e}")
            return False

    def open_short(self, price: float) -> bool:
        size = round(MARGIN_PER_TRADE * LEVERAGE / price, 5)
        if size < 0.001:
            print(f"[WARN] Size {size} too small for {COIN}")
            return False
        try:
            result = self.exchange.market_open(COIN, False, size, None, 0.01)
            print(f"[OPEN SHORT] size={size} @ ~{price}")
            return True
        except Exception as e:
            print(f"[ERROR] open_short failed: {e}")
            return False

    def open_long(self, price: float) -> bool:
        size = round(MARGIN_PER_TRADE * LEVERAGE / price, 5)
        if size < 0.001:
            print(f"[WARN] Size {size} too small for {COIN}")
            return False
        try:
            result = self.exchange.market_open(COIN, True, size, None, 0.01)
            print(f"[OPEN LONG] size={size} @ ~{price}")
            return True
        except Exception as e:
            print(f"[ERROR] open_long failed: {e}")
            return False

    def close_position(self, reason: str = "manual") -> bool:
        try:
            pos = self.get_position()
            if pos is None:
                return True
            szi = float(pos.get("szi", 0))
            if abs(szi) < 0.0001:
                return True
            is_buy = szi < 0
            result = self.exchange.market_close(COIN)
            print(f"[CLOSE] reason={reason}")
            return True
        except Exception as e:
            print(f"[ERROR] close failed: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────
# Strategy logic
# ──────────────────────────────────────────────────────────────────────

def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def detect_trend(prices: list[float]) -> str:
    ema_f = ema(prices, EMA_FAST)
    ema_s = ema(prices, EMA_SLOW)
    if ema_f is None or ema_s is None:
        return "neutral"
    if ema_f < ema_s:
        return "bear"
    if ema_f > ema_s:
        return "bull"
    return "neutral"


def compute_atr(prices: list[float], period: int = 20) -> float | None:
    if len(prices) < period + 1:
        return None
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    recent = changes[-(period):]
    return float(np.mean(recent)) if recent else None


def should_pause_volatility(prices: list[float]) -> tuple[bool, str]:
    atr_now = compute_atr(prices, 20)
    if atr_now is None or len(prices) < 120:
        return False, ""
    long_atr = compute_atr(prices, min(120, len(prices) - 1))
    if long_atr is None or long_atr < 1:
        return False, ""
    ratio = atr_now / long_atr
    if ratio > VOLATILITY_PAUSE_ATR_MULT:
        return True, f"volatility spike: short_atr/long_atr={ratio:.2f}"
    return False, ""


def check_daily_reset(state: BotState) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.daily_reset_ts != today:
        state.daily_trades = 0
        state.daily_pnl = 0.0
        state.daily_reset_ts = today
        print(f"[DAILY RESET] New trading day: {today}")


def check_risk_limits(state: BotState) -> tuple[bool, str]:
    if state.daily_pnl <= -MAX_DAILY_LOSS_USD:
        return False, f"Daily loss limit reached: ${state.daily_pnl:.2f}"
    if state.daily_trades >= MAX_DAILY_TRADES:
        return False, f"Daily trade limit reached: {state.daily_trades}"
    return True, ""


def manage_open_position(bot: HyperliquidBot, state: BotState, current_price: float) -> bool:
    pos = bot.get_position()
    if pos is None or float(pos.get("szi", 0)) == 0:
        state.open_position = None
        return False

    szi = float(pos["szi"])
    entry_price = float(pos.get("entryPx", current_price))
    side = "short" if szi < 0 else "long"
    unrealized_pnl = float(pos.get("unrealizedPnl", 0))

    tp_price = entry_price - TAKE_PROFIT_POINTS if side == "short" else entry_price + TAKE_PROFIT_POINTS
    sl_price = entry_price + STOP_LOSS_POINTS if side == "short" else entry_price - STOP_LOSS_POINTS

    if side == "short":
        if current_price <= tp_price:
            bot.close_position("take_profit")
            _record_trade(state, side, entry_price, current_price, abs(szi), unrealized_pnl, "tp")
            return True
        if current_price >= sl_price:
            bot.close_position("stop_loss")
            _record_trade(state, side, entry_price, current_price, abs(szi), unrealized_pnl, "sl")
            return True
    else:
        if current_price >= tp_price:
            bot.close_position("take_profit")
            _record_trade(state, side, entry_price, current_price, abs(szi), unrealized_pnl, "tp")
            return True
        if current_price <= sl_price:
            bot.close_position("stop_loss")
            _record_trade(state, side, entry_price, current_price, abs(szi), unrealized_pnl, "sl")
            return True

    open_minutes = (time.time() - state.last_trade_ts) / 60.0
    if open_minutes > POSITION_TIMEOUT_MINUTES:
        bot.close_position("timeout")
        _record_trade(state, side, entry_price, current_price, abs(szi), unrealized_pnl, "timeout")
        return True

    state.open_position = {"side": side, "entry": entry_price, "pnl": unrealized_pnl,
                           "minutes_open": open_minutes}
    return False


def try_open_trade(bot: HyperliquidBot, state: BotState, current_price: float, trend: str) -> None:
    if trend == "bear":
        recent_high = max(state.price_history) if state.price_history else current_price
        pullback = current_price - recent_high
        if pullback <= -ENTRY_PULLBACK_POINTS:
            print(f"[SIGNAL] Bear regime, price bounced up {ENTRY_PULLBACK_POINTS}pts → short")
            if bot.open_short(current_price):
                state.daily_trades += 1
                state.last_trade_ts = time.time()

    elif trend == "bull":
        recent_low = min(state.price_history) if state.price_history else current_price
        pullback = recent_low - current_price
        if pullback >= ENTRY_PULLBACK_POINTS:
            print(f"[SIGNAL] Bull regime, price dipped {ENTRY_PULLBACK_POINTS}pts → long")
            if bot.open_long(current_price):
                state.daily_trades += 1
                state.last_trade_ts = time.time()


def _record_trade(state: BotState, side: str, entry: float, exit_: float,
                  size: float, pnl: float, reason: str) -> None:
    state.daily_pnl += pnl
    record = TradeRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        side=side, entry_price=entry, exit_price=exit_,
        size=size, pnl=pnl, exit_reason=reason,
        duration_seconds=time.time() - state.last_trade_ts,
    )
    state.trade_history.append(record)
    print(f"[TRADE CLOSED] {side} entry={entry:.1f} exit={exit_:.1f} "
          f"pnl=${pnl:+.4f} reason={reason} "
          f"duration={record.duration_seconds:.0f}s")
    print(f"[DAILY] trades={state.daily_trades} pnl=${state.daily_pnl:.2f}")


def save_logs(state: BotState) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"trades_{today}.json"
    records = [
        {
            "timestamp": r.timestamp, "side": r.side,
            "entry": r.entry_price, "exit": r.exit_price,
            "size": r.size, "pnl": r.pnl, "reason": r.exit_reason,
            "duration_s": round(r.duration_seconds, 1),
        }
        for r in state.trade_history
    ]
    with open(log_file, "w") as f:
        json.dump(records, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Hyperliquid BTC Scalper")
    print("=" * 60)
    print(f"  Coin:        {COIN}")
    print(f"  Leverage:    {LEVERAGE}x")
    print(f"  Margin/trade: ${MARGIN_PER_TRADE}")
    print(f"  TP:          {TAKE_PROFIT_POINTS} pts")
    print(f"  SL:          {STOP_LOSS_POINTS} pts")
    print(f"  EMA:         fast={EMA_FAST} slow={EMA_SLOW}")
    print(f"  Max daily loss: ${MAX_DAILY_LOSS_USD}")
    print(f"  Max daily trades: {MAX_DAILY_TRADES}")
    print(f"  Poll interval: {POLL_INTERVAL_SECONDS}s")
    print("=" * 60)

    bot = HyperliquidBot()
    bot.set_leverage()

    state = BotState()
    state.last_trade_ts = time.time()
    check_daily_reset(state)

    existing = bot.get_position()
    if existing and float(existing.get("szi", 0)) != 0:
        print(f"\n⚠️  Existing position detected: {existing}")
        print("Bot will manage this position (monitor TP/SL) before opening new ones.")

    def shutdown(sig, frame):
        print("\n[SHUTDOWN] Stopping bot gracefully...")
        state.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\nBot started. Press Ctrl+C to stop.\n")

    loop_count = 0
    while state.running:
        try:
            loop_count += 1
            check_daily_reset(state)

            price = bot.get_price()
            if price is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            state.price_history.append(price)

            if time.time() - state.last_candle_refresh > TREND_REFRESH_SECONDS:
                closes = bot.get_candles(TREND_TIMEFRAME, max(EMA_SLOW + 20, 200))
                if len(closes) >= EMA_SLOW:
                    state.candle_closes = closes
                    state.current_trend = detect_trend(closes)
                    state.last_candle_refresh = time.time()
                    ema_f = ema(closes, EMA_FAST)
                    ema_s = ema(closes, EMA_SLOW)
                    print(f"[TREND UPDATE] {TREND_TIMEFRAME} trend={state.current_trend} "
                          f"EMA{EMA_FAST}={ema_f:.1f} EMA{EMA_SLOW}={ema_s:.1f} "
                          f"price={price:.1f} candles={len(closes)}")

            closed = manage_open_position(bot, state, price)

            if state.open_position is not None and not closed:
                pos = state.open_position
                if loop_count % 12 == 0:
                    print(f"[MONITOR] price={price:.1f} "
                          f"pos={pos['side']} entry={pos['entry']:.1f} "
                          f"uPnL=${pos['pnl']:+.4f} "
                          f"open={pos['minutes_open']:.1f}min "
                          f"trend={state.current_trend}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            can_trade, reason = check_risk_limits(state)
            if not can_trade:
                if loop_count % 60 == 0:
                    print(f"[PAUSED] {reason}")
                time.sleep(POLL_INTERVAL_SECONDS * 2)
                continue

            prices_list = list(state.price_history)
            pause, vol_reason = should_pause_volatility(prices_list)
            if pause:
                if loop_count % 60 == 0:
                    print(f"[VOLATILITY PAUSE] {vol_reason}")
                time.sleep(POLL_INTERVAL_SECONDS * 2)
                continue

            try_open_trade(bot, state, price, state.current_trend)

            if loop_count % 60 == 0:
                print(f"[STATUS] price={price:.1f} trend={state.current_trend} "
                      f"trades={state.daily_trades} pnl=${state.daily_pnl:.2f} "
                      f"prices_buffered={len(state.price_history)}")

            save_logs(state)
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(10)

    save_logs(state)
    print(f"\nBot stopped. Today: {state.daily_trades} trades, PnL: ${state.daily_pnl:.2f}")
    print(f"Total trades logged: {len(state.trade_history)}")


if __name__ == "__main__":
    main()
