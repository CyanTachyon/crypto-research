"""Eight frozen, causal research hypotheses. No exchange or account access.

Every output at index i uses only candles at or before i. The execution engine
must wait until candle i closes and execute no earlier than candle i + 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


CANDIDATES = (
    "breakout_72", "breakout_168", "breakout_336",
    "momentum_72", "momentum_168", "momentum_336",
    "ema_24_120", "range_48",
)


@dataclass(frozen=True)
class Signal:
    direction: int = 0
    strength: float = 0.0
    atr: float = 0.0
    exit_long: bool = False
    exit_short: bool = False


def _ema(values: list[float], period: int) -> list[float | None]:
    """SMA-seeded EMA, unavailable until a full seed window exists."""
    output: list[float | None] = [None] * len(values)
    if len(values) < period:
        return output
    output[period - 1] = sum(values[:period]) / period
    alpha = 2.0 / (period + 1)
    for i in range(period, len(values)):
        output[i] = alpha * values[i] + (1 - alpha) * output[i - 1]
    return output


def _atr(candles: list[Mapping], period: int = 24) -> list[float]:
    """Arithmetic mean of true ranges, including the just-completed candle."""
    tr = []
    output = [0.0] * len(candles)
    for i, candle in enumerate(candles):
        high, low = float(candle["h"]), float(candle["l"])
        prev = float(candles[i - 1]["c"]) if i else float(candle["o"])
        tr.append(max(high - low, abs(high - prev), abs(low - prev)))
        if i >= period - 1:
            output[i] = sum(tr[i - period + 1:i + 1]) / period
    return output


def _validate(candles: list[Mapping], coin: str) -> None:
    previous_t = None
    for row in candles:
        prices = [float(row[key]) for key in ("o", "h", "l", "c")]
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise ValueError(f"{coin}: non-finite or non-positive OHLC")
        op, high, low, close = prices
        if high < max(op, low, close) or low > min(op, high, close):
            raise ValueError(f"{coin}: inconsistent OHLC")
        if previous_t is not None and row["t"] <= previous_t:
            raise ValueError(f"{coin}: candle timestamps must strictly increase")
        previous_t = row["t"]


def build_signals(dataset: Mapping) -> dict[str, dict[str, list[Signal]]]:
    """Accept the hl20 schema with assets[coin].candles of {t,o,h,l,c,v}.

    Neutral trend/breakout signals mean no new entry, not an exit. Range exits
    are explicit and remain active when the flat-trend entry filter fails.
    The engine owns position state, risk sizing, cooldowns and the 336h warmup.
    """
    assets = dataset["assets"]
    if set(assets) != {"BTC", "ETH"}:
        raise ValueError("Research universe must contain exactly BTC and ETH")
    result = {name: {} for name in CANDIDATES}
    for coin in ("BTC", "ETH"):
        candles = assets[coin]["candles"]
        _validate(candles, coin)
        closes = [float(row["c"]) for row in candles]
        atr = _atr(candles)
        fast, slow = _ema(closes, 24), _ema(closes, 120)
        for name in CANDIDATES:
            result[name][coin] = [Signal(atr=value) for value in atr]
        for i, close in enumerate(closes):
            if atr[i] <= 0:
                continue
            for period in (72, 168, 336):
                if i < period:
                    continue
                # Deliberately exclude current candle from channel extrema.
                previous = candles[i - period:i]
                upper = max(float(row["h"]) for row in previous)
                lower = min(float(row["l"]) for row in previous)
                direction = 1 if close > upper else -1 if close < lower else 0
                strength = abs(close - (upper + lower) / 2) / atr[i] if direction else 0.0
                result[f"breakout_{period}"][coin][i] = Signal(direction, strength, atr[i])
                change = close - closes[i - period]
                direction = 1 if change > 0 else -1 if change < 0 else 0
                result[f"momentum_{period}"][coin][i] = Signal(direction, abs(change) / atr[i], atr[i])
            if fast[i] is not None and slow[i] is not None:
                change = fast[i] - slow[i]
                direction = 1 if change > 0 else -1 if change < 0 else 0
                result["ema_24_120"][coin][i] = Signal(direction, abs(change) / atr[i], atr[i])
            if i >= 143:  # EMA120 plus a full 24h slope comparison.
                window = closes[i - 47:i + 1]
                mean = sum(window) / 48
                variance = sum((value - mean) ** 2 for value in window) / 48
                std = math.sqrt(variance)
                z = (close - mean) / std if std > 0 else 0.0
                flat = abs(slow[i] - slow[i - 24]) / atr[i] <= 0.5
                direction = 1 if flat and z <= -1.5 else -1 if flat and z >= 1.5 else 0
                result["range_48"][coin][i] = Signal(
                    direction, abs(z) if direction else 0.0, atr[i],
                    exit_long=close >= mean, exit_short=close <= mean,
                )
    return result
