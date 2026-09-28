"""Causality and strategy invariants, independent of backtest profitability."""

import copy
import math
import unittest

from strategies import CANDIDATES, _atr, _ema, build_signals


def dataset(count=450, curve=None):
    curve = curve or (lambda i: 100 + i * .03 + math.sin(i / 13))
    rows = []
    for i in range(count):
        close = curve(i)
        rows.append({"t": i * 3_600_000, "o": close, "h": close + .5,
                     "l": close - .5, "c": close, "v": 1})
    return {"assets": {coin: {"candles": copy.deepcopy(rows)} for coin in ("BTC", "ETH")}}


class StrategyTests(unittest.TestCase):
    def test_exact_frozen_universe_and_candidate_count(self):
        data = dataset()
        signals = build_signals(data)
        self.assertEqual(tuple(signals), CANDIDATES)
        self.assertEqual(len(signals), 8)
        for assets in signals.values():
            self.assertEqual(set(assets), {"BTC", "ETH"})
            for rows in assets.values():
                self.assertEqual(len(rows), 450)
                self.assertTrue(all(s.direction in (-1, 0, 1) and s.strength >= 0 and s.atr >= 0 for s in rows))

    def test_future_changes_cannot_change_past_signals(self):
        original = dataset()
        changed = copy.deepcopy(original)
        for asset in changed["assets"].values():
            for row in asset["candles"][370:]:
                for key in ("o", "h", "l", "c"):
                    row[key] *= 2
        first, second = build_signals(original), build_signals(changed)
        prefix = copy.deepcopy(original)
        for asset in prefix["assets"].values():
            asset["candles"] = asset["candles"][:370]
        prefix_signals = build_signals(prefix)
        for name in CANDIDATES:
            for coin in ("BTC", "ETH"):
                self.assertEqual(first[name][coin][:370], second[name][coin][:370])
                self.assertEqual(first[name][coin][:370], prefix_signals[name][coin])

    def test_breakout_channel_excludes_current_candle(self):
        data = dataset(74, lambda _: 100)
        for asset in data["assets"].values():
            asset["candles"][72].update(o=101, h=101.5, l=100.5, c=101)
        output = build_signals(data)
        self.assertEqual(output["breakout_72"]["BTC"][71].direction, 0)
        self.assertEqual(output["breakout_72"]["BTC"][72].direction, 1)
        self.assertEqual(output["breakout_72"]["BTC"][73].direction, 0)

    def test_momentum_and_ema_follow_direction_after_warmup(self):
        output = build_signals(dataset(400, lambda i: 100 + i))
        for name in ("momentum_72", "momentum_168", "momentum_336", "ema_24_120"):
            self.assertEqual(output[name]["BTC"][-1].direction, 1)
        output = build_signals(dataset(400, lambda i: 500 - i))
        for name in ("momentum_72", "momentum_168", "momentum_336", "ema_24_120"):
            self.assertEqual(output[name]["BTC"][-1].direction, -1)

    def test_range_has_separate_entry_and_exit_flags(self):
        data = dataset(160, lambda _: 100)
        for asset in data["assets"].values():
            asset["candles"][158].update(o=99, h=99.5, l=98.5, c=99)
        output = build_signals(data)["range_48"]["BTC"]
        self.assertEqual(output[158].direction, 1)
        self.assertFalse(output[158].exit_long)
        self.assertTrue(output[159].exit_long)

    def test_atr_includes_gap_and_ema_seed_is_sma(self):
        data = dataset(24, lambda _: 100)
        data["assets"]["BTC"]["candles"][23].update(o=110, h=111, l=109, c=110)
        atr = _atr(data["assets"]["BTC"]["candles"])
        self.assertEqual(atr[22], 0)
        self.assertAlmostEqual(atr[23], (23 + 11) / 24)
        self.assertEqual(_ema([1, 2, 3, 4], 3), [None, None, 2, 3])

    def test_invalid_prices_and_universe_fail_closed(self):
        data = dataset(30)
        data["assets"]["BTC"]["candles"][5]["c"] = float("nan")
        with self.assertRaises(ValueError):
            build_signals(data)
        data = dataset(30)
        data["assets"]["SOL"] = data["assets"]["ETH"]
        with self.assertRaises(ValueError):
            build_signals(data)


if __name__ == "__main__":
    unittest.main()
