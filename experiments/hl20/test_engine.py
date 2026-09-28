"""Adversarial numerical tests; these do not certify real exchange execution.

Run: python3 -m unittest discover -s experiments/hl20 -p 'test_*.py' -v
"""
import copy
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from engine import Config, Engine, HOUR, floor_size, run_backtest


def signal(direction=1, atr=1.0, strength=1.0, **kwargs):
    return SimpleNamespace(direction=direction, atr=atr, strength=strength,
                           exit_long=kwargs.get("exit_long", False),
                           exit_short=kwargs.get("exit_short", False))


def bars(hour=0, price=100.0, high=None, low=None, close=None):
    row = dict(t=hour * HOUR, o=price, h=price if high is None else high,
               l=price if low is None else low, c=price if close is None else close)
    return {coin: dict(row) for coin in ("BTC", "ETH")}


ZERO_FUNDING = {"BTC": 0.0, "ETH": 0.0}


class EngineTests(unittest.TestCase):
    def engine(self, **overrides):
        config = replace(Config(), fee_bps=0, slippage_bps=0, stop_atr=1)
        return Engine(replace(config, **overrides), {"BTC": 5, "ETH": 4})

    def enter(self, direction=1, **overrides):
        engine = self.engine(**overrides)
        engine.step(bars(), {"BTC": signal(direction)}, ZERO_FUNDING)
        self.assertIsNotNone(engine.position)
        return engine

    def assert_reconciled(self, engine):
        report = engine.report()
        self.assertAlmostEqual(engine.cash, 20 + report["realized_gross"]
                               - report["fees"] - report["funding_paid"], places=11)
        if engine.position is None:
            self.assertAlmostEqual(engine.cash - 20,
                                   sum(t["net"] for t in report["trades"]), places=11)

    def test_both_fees_match_cash_and_trade_net(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                engine = self.enter(direction, fee_bps=4.5)
                self.assertAlmostEqual(engine.position.size, .15)
                engine.finish()
                self.assertAlmostEqual(engine.report()["fees"], .0135)
                self.assertAlmostEqual(engine.cash, 19.9865)
                self.assertAlmostEqual(engine.trades[0]["net"], -.0135)
                self.assert_reconciled(engine)

    def test_short_profit_has_correct_sign_and_exit_fee(self):
        engine = self.enter(-1, fee_bps=4.5)
        engine.step(bars(1, price=90), {}, ZERO_FUNDING)
        engine.finish()
        self.assertAlmostEqual(engine.trades[0]["gross"], 1.5)
        self.assertAlmostEqual(engine.trades[0]["fees"], .012825)
        self.assertAlmostEqual(engine.trades[0]["net"], 1.487175)
        self.assert_reconciled(engine)

    def test_slippage_always_costs_on_flat_market(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                engine = self.enter(direction, slippage_bps=10)
                size = engine.position.size
                self.assertAlmostEqual(engine.position.entry, 100 + direction * .1)
                engine.finish()
                self.assertAlmostEqual(engine.trades[0]["exit"], 100 - direction * .1)
                self.assertAlmostEqual(engine.cash - 20, -.2 * size)
                self.assertLess(engine.cash, 20)
                self.assert_reconciled(engine)

    def test_funding_all_direction_and_rate_combinations(self):
        for direction in (-1, 1):
            for rate in (-.001, .001):
                with self.subTest(direction=direction, rate=rate):
                    engine = self.enter(direction)
                    engine.step(bars(1), {}, {"BTC": rate, "ETH": .9})
                    self.assertAlmostEqual(engine.funding_paid, direction * 15 * rate)
                    engine.finish()
                    self.assertAlmostEqual(engine.cash - 20, -direction * 15 * rate)
                    self.assert_reconciled(engine)

    def test_funding_uses_held_quantity_and_current_boundary_price(self):
        engine = self.enter(-1)
        engine.step(bars(1, price=90), {}, {"BTC": .001, "ETH": 0})
        self.assertAlmostEqual(engine.funding_paid, -.0135)

    def test_no_funding_before_entry_at_same_boundary(self):
        engine = self.engine()
        engine.step(bars(), {"BTC": signal()}, {"BTC": .5, "ETH": .5})
        self.assertAlmostEqual(engine.cash, 20)
        self.assertAlmostEqual(engine.position.funding_paid, 0)

    def test_funding_is_paid_before_boundary_exit(self):
        engine = self.enter()
        engine.step(bars(1), {"BTC": signal(-1)}, {"BTC": .001, "ETH": 0})
        self.assertIsNone(engine.position)
        self.assertAlmostEqual(engine.trades[0]["funding_paid"], .015)
        self.assertAlmostEqual(engine.trades[0]["net"], -.015)

    def test_funding_stress_doubles_cost_and_removes_income(self):
        for direction, expected in ((1, .03), (-1, 0)):
            with self.subTest(direction=direction):
                engine = self.enter(direction, funding_cost_multiplier=2,
                                    funding_credit_fraction=0)
                engine.step(bars(1), {}, {"BTC": .001, "ETH": 0})
                self.assertAlmostEqual(engine.funding_paid, expected)

    def test_floor_size_never_rounds_up(self):
        self.assertEqual(floor_size(.00001999999, 5), .00001)
        self.assertEqual(floor_size(.1234999999, 4), .1234)
        self.assertEqual(floor_size(.999, 0), 0)
        for size, precision in ((-1, 5), (math.nan, 5), (1, 7)):
            with self.assertRaises(ValueError):
                floor_size(size, precision)

    def test_minimum_order_applies_after_quantity_floor(self):
        engine = Engine(replace(Config(), max_notional=10, stop_atr=1,
                                fee_bps=0, slippage_bps=0), {"BTC": 0, "ETH": 4})
        engine.step(bars(price=3), {"BTC": signal(atr=.01)}, ZERO_FUNDING)
        self.assertIsNone(engine.position)  # raw 10 USD becomes 3 * 3 = 9 USD.
        self.assertEqual(engine.skips["below_min_notional"], 1)
        self.assertEqual(engine.cash, 20)

    def test_size_below_minimum_is_skipped_instead_of_increased(self):
        engine = self.engine()
        engine.step(bars(), {"BTC": signal(atr=3)}, ZERO_FUNDING)
        self.assertIsNone(engine.position)
        self.assertEqual(engine.skips["below_min_notional"], 1)
        self.assertEqual(engine.fees, 0)

    def test_zero_quantity_is_not_a_position(self):
        engine = Engine(replace(Config(), stop_atr=1), {"BTC": 0, "ETH": 4})
        engine.step(bars(price=100000), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNone(engine.position)

    def test_exact_planned_stop_loss_includes_short_exit_friction(self):
        engine = self.engine(fee_bps=4.5, slippage_bps=2)
        engine.step(bars(), {"BTC": signal(-1, atr=1.399)}, ZERO_FUNDING)
        engine.step(bars(1, high=102), {}, ZERO_FUNDING)
        self.assertLessEqual(-engine.trades[0]["net"], .2 + 1e-12)
        self.assertGreater(-engine.trades[0]["net"], .199)
        self.assert_reconciled(engine)

    def test_single_position_no_addition_or_coin_switch(self):
        engine = self.enter()
        initial = copy.deepcopy(engine.position)
        engine.step(bars(1), {"BTC": signal(), "ETH": signal(strength=10)}, ZERO_FUNDING)
        self.assertEqual(engine.position.coin, "BTC")
        self.assertEqual(engine.position.size, initial.size)
        self.assertEqual(engine.position.entry_time, initial.entry_time)
        self.assertLessEqual(engine.position.size * engine.position.entry, 15)

    def test_equal_strength_prefers_btc_as_frozen_in_protocol(self):
        engine = self.engine()
        engine.step(bars(), {"BTC": signal(), "ETH": signal()}, ZERO_FUNDING)
        self.assertEqual(engine.position.coin, "BTC")

    def test_gap_stop_uses_worse_open_both_directions(self):
        for direction, price in ((1, 95), (-1, 105)):
            with self.subTest(direction=direction):
                engine = self.enter(direction)
                engine.step(bars(1, price=price), {"BTC": signal(direction)}, ZERO_FUNDING)
                self.assertIsNone(engine.position)
                self.assertEqual(engine.trades[0]["exit_reason"], "stop_gap")
                self.assertAlmostEqual(engine.trades[0]["exit"], price)
                self.assertAlmostEqual(engine.trades[0]["net"], -.75)
                self.assert_reconciled(engine)

    def test_intrabar_stop_closes_even_when_close_recovers(self):
        for direction, high, low in ((1, 102, 98), (-1, 102, 98)):
            with self.subTest(direction=direction):
                engine = self.enter(direction)
                engine.step(bars(1, high=high, low=low), {"BTC": signal(direction)}, ZERO_FUNDING)
                self.assertIsNone(engine.position)
                self.assertEqual(engine.trades[0]["exit_reason"], "stop")
                self.assertAlmostEqual(engine.trades[0]["net"], -.15)
                self.assertEqual(engine.trades[0]["exit_time"], 2 * HOUR - 1)

    def test_stop_on_entry_bar_is_executable(self):
        engine = self.engine()
        engine.step(bars(low=98, close=101, high=101), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNone(engine.position)
        self.assertAlmostEqual(engine.cash, 19.85)
        self.assertEqual(len(engine.trades), 1)

    def test_completed_close_trailing_stop_is_only_effective_next_hour(self):
        engine = self.enter()
        engine.step(bars(1, high=110, low=100, close=110), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNotNone(engine.position)
        self.assertAlmostEqual(engine.position.stop, 109)
        engine.step(bars(2, price=110, low=108), {}, ZERO_FUNDING)
        self.assertAlmostEqual(engine.trades[0]["exit"], 109)

    def test_completed_current_atr_only_changes_next_hours_trailing_stop(self):
        engine = self.enter()
        engine.step(bars(1, high=110, low=100, close=110), {"BTC": signal(atr=1)},
                    ZERO_FUNDING, trailing_atrs={"BTC": 3, "ETH": 3})
        self.assertIsNotNone(engine.position)  # New stop cannot reach backward into this bar.
        self.assertAlmostEqual(engine.position.stop, 107)  # Uses current ATR=3, not prior ATR=1.
        engine.step(bars(2, price=110, low=108), {}, ZERO_FUNDING,
                    trailing_atrs={"BTC": 3, "ETH": 3})
        self.assertIsNotNone(engine.position)
        engine.step(bars(3, price=110, low=106), {}, ZERO_FUNDING)
        self.assertAlmostEqual(engine.trades[0]["exit"], 107)

    def test_cooldown_prevents_repeat_entry_after_intrabar_stop(self):
        engine = self.enter()
        engine.step(bars(1, low=98), {"BTC": signal()}, ZERO_FUNDING)
        for hour in range(2, 6):
            engine.step(bars(hour), {"BTC": signal()}, ZERO_FUNDING)
            self.assertIsNone(engine.position)
        engine.step(bars(6), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNotNone(engine.position)

    def test_reverse_closes_without_same_bar_flip(self):
        engine = self.enter()
        engine.step(bars(1), {"BTC": signal(-1)}, ZERO_FUNDING)
        self.assertIsNone(engine.position)
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.trades[0]["exit_reason"], "signal_exit")

    def test_neutral_signal_does_not_exit_trend_position(self):
        engine = self.enter()
        engine.step(bars(1), {"BTC": signal(0)}, ZERO_FUNDING)
        self.assertIsNotNone(engine.position)

    def test_explicit_mean_exit_closes_without_opposite_signal(self):
        engine = self.enter()
        engine.step(bars(1), {"BTC": signal(0, exit_long=True)}, ZERO_FUNDING)
        self.assertIsNone(engine.position)
        self.assertEqual(engine.trades[0]["exit_reason"], "signal_exit")

    def test_timeout_checked_without_decision_boundary(self):
        engine = self.enter(max_hold_hours=1)
        engine.step(bars(1), {}, ZERO_FUNDING, decision=False)
        self.assertIsNone(engine.position)
        self.assertEqual(engine.trades[0]["exit_reason"], "time_exit")

    def test_drawdown_halt_uses_highwater_and_cannot_auto_resume(self):
        engine = self.enter()
        engine.step(bars(1, price=120), {}, ZERO_FUNDING)
        self.assertAlmostEqual(engine.peak, 23)
        engine.step(bars(2, price=100), {"BTC": signal()}, ZERO_FUNDING)
        self.assertTrue(engine.halted)
        self.assertIsNone(engine.position)
        self.assertEqual(engine.halt_reason, "account_drawdown")
        self.assertAlmostEqual(engine.cash, 20)
        for hour in range(3, 30):
            engine.step(bars(hour), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNone(engine.position)
        self.assertEqual(len(engine.trades), 1)

    def test_gap_loss_is_not_clipped_to_account_threshold(self):
        engine = self.enter()
        engine.step(bars(1, price=50), {}, ZERO_FUNDING)
        self.assertTrue(engine.halted)
        self.assertAlmostEqual(engine.cash, 12.5)
        self.assertAlmostEqual(engine.report()["max_drawdown_pct"], 37.5)

    def test_intrabar_account_stop_when_tighter_than_position_stop(self):
        engine = self.enter()
        engine.step(bars(1, price=120, high=121, low=100), {}, ZERO_FUNDING)
        self.assertTrue(engine.halted)
        self.assertEqual(engine.trades[0]["exit_reason"], "account_drawdown")
        self.assertAlmostEqual(engine.cash, 20.7)

    def test_daily_loss_blocks_entry_until_next_utc_day(self):
        engine = self.enter()
        engine.step(bars(1, price=95), {}, ZERO_FUNDING)
        for hour in range(2, 24):
            engine.step(bars(hour), {"BTC": signal()}, ZERO_FUNDING)
            self.assertIsNone(engine.position)
        self.assertGreater(engine.skips.get("daily_loss_limit", 0), 0)
        engine.step(bars(24), {"BTC": signal()}, ZERO_FUNDING)
        self.assertIsNotNone(engine.position)

    def test_daily_loss_latches_even_if_held_position_recovers(self):
        engine = self.enter()
        engine.step(bars(1), {}, {"BTC": .05, "ETH": 0})
        self.assertAlmostEqual(engine.cash, 19.25)
        self.assertIsNotNone(engine.position)  # Existing risk management continues.
        engine.step(bars(2, price=105), {"BTC": signal(0, exit_long=True)}, ZERO_FUNDING)
        self.assertAlmostEqual(engine.cash, 20)
        for hour in range(3, 24):
            engine.step(bars(hour), {"BTC": signal()}, ZERO_FUNDING)
            self.assertIsNone(engine.position)

    def test_daily_loss_latches_on_observed_intrabar_adverse_price(self):
        engine = self.enter()
        engine.step(bars(1, low=99.85), {}, {"BTC": .59 / 15, "ETH": 0})
        self.assertAlmostEqual(engine.cash, 19.41)
        self.assertIsNotNone(engine.position)
        engine.step(bars(2, price=101), {"BTC": signal(0, exit_long=True)}, ZERO_FUNDING)
        for hour in range(3, 24):
            engine.step(bars(hour), {"BTC": signal()}, ZERO_FUNDING)
            self.assertIsNone(engine.position)

    def test_multiple_completed_trades_reconcile_without_duplicate_entry_fees(self):
        engine = self.engine(fee_bps=4.5, slippage_bps=2, cooldown_hours=0)
        for hour in range(40):
            if hour % 2 == 0:
                direction = 1 if hour % 4 == 0 else -1
                sig = signal(direction)
            else:
                sig = signal(0, exit_long=True, exit_short=True)
            engine.step(bars(hour), {"BTC": sig}, {"BTC": .0001, "ETH": 0})
            self.assert_reconciled(engine)
        self.assertEqual(len(engine.trades), 20)
        self.assertAlmostEqual(engine.fees, sum(t["fees"] for t in engine.trades))
        self.assertAlmostEqual(engine.funding_paid, sum(t["funding_paid"] for t in engine.trades))

    def test_invalid_price_time_and_funding_are_rejected_before_mutation(self):
        bad_cases = []
        for key, value in (("o", math.nan), ("c", 0), ("h", 90), ("l", 110)):
            bad = bars()
            bad["BTC"][key] = value
            bad_cases.append((bad, ZERO_FUNDING))
        bad = bars()
        bad["ETH"]["t"] = HOUR
        bad_cases.append((bad, ZERO_FUNDING))
        bad_cases.append((bars(), {"BTC": math.inf, "ETH": 0}))
        bad_cases.append((bars(), {"BTC": 0}))
        for bad, funding in bad_cases:
            with self.subTest(bad=bad, funding=funding):
                engine = self.engine()
                with self.assertRaises(ValueError):
                    engine.step(bad, {"BTC": signal()}, funding)
                self.assertEqual(engine.cash, 20)
                self.assertIsNone(engine.position)

    def test_duplicate_and_missing_hour_cannot_silently_replay(self):
        for hour in (0, 2):
            engine = self.enter()
            with self.assertRaises(ValueError):
                engine.step(bars(hour), {}, ZERO_FUNDING)

    def test_finish_is_idempotent_and_final_curve_equals_cash(self):
        engine = self.enter(fee_bps=4.5)
        engine.finish()
        final_cash = engine.cash
        engine.finish()
        self.assertEqual(engine.cash, final_cash)
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.equity_curve[-1]["equity"], final_cash)
        self.assertEqual(engine.equity_curve[-1]["gross_notional"], 0)


class BacktestTimingTests(unittest.TestCase):
    def fixture(self):
        data = {"assets": {}}
        sig = {}
        for coin in ("BTC", "ETH"):
            data["assets"][coin] = {"sz_decimals": 5 if coin == "BTC" else 4,
                                     "candles": [bars(i)[coin] for i in range(9)],
                                     "funding": [{"t": i * HOUR, "rate": 0} for i in range(9)]}
            sig[coin] = [signal(0) for _ in range(9)]
        return data, sig

    def config(self):
        return replace(Config(), fee_bps=0, slippage_bps=0, stop_atr=1)

    def test_signal_from_prior_bar_executes_at_next_eligible_open(self):
        data, sig = self.fixture()
        sig["BTC"][3] = signal()
        data["assets"]["BTC"]["candles"][4] = bars(4, price=101)["BTC"]
        data["assets"]["BTC"]["candles"][5] = bars(5, price=101)["BTC"]
        result = run_backtest(data, sig, self.config(), 1, 6)
        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["trades"][0]["entry_time"], 4 * HOUR)
        self.assertEqual(result["trades"][0]["entry"], 101)

    def test_current_bar_signal_cannot_trade_its_own_open(self):
        data, sig = self.fixture()
        sig["BTC"][4] = signal()
        result = run_backtest(data, sig, self.config(), 1, 5)
        self.assertEqual(result["trade_count"], 0)

    def test_runner_passes_current_atr_for_close_only_trailing(self):
        data, sig = self.fixture()
        sig["BTC"][3] = signal(atr=1)
        sig["BTC"][4] = signal(0, atr=3)
        sig["BTC"][5] = signal(0, atr=3)
        data["assets"]["BTC"]["candles"][4] = bars(4, price=100, high=110, close=110)["BTC"]
        data["assets"]["BTC"]["candles"][5] = bars(5, price=110, low=108)["BTC"]
        data["assets"]["BTC"]["candles"][6] = bars(6, price=110, low=106)["BTC"]
        result = run_backtest(data, sig, self.config(), 1, 7)
        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["trades"][0]["exit_time"], 7 * HOUR - 1)
        self.assertEqual(result["trades"][0]["exit"], 107)

    def test_zero_delay_is_explicitly_rejected(self):
        data, sig = self.fixture()
        with self.assertRaises(ValueError):
            run_backtest(data, sig, self.config(), 1, 5, decision_delay=0)

    def test_missing_and_duplicate_funding_fail_instead_of_becoming_zero(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                data, sig = self.fixture()
                if duplicate:
                    data["assets"]["BTC"]["funding"].append({"t": 3 * HOUR, "rate": 0})
                else:
                    data["assets"]["BTC"]["funding"].pop(3)
                with self.assertRaises(ValueError):
                    run_backtest(data, sig, self.config(), 1, 5)


if __name__ == "__main__":
    unittest.main()
