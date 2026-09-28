"""Offline integrity/retry/pagination tests; never query accounts or place orders."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from data_source import (ASSETS, HOUR_MS as H, DataIntegrityError, PublicInfoClient,
                         fetch_funding, load_verified, normalize_candles, save_dataset,
                         validate_dataset)


def fixture():
    rows = [{"t": i * H, "o": 100., "h": 110., "l": 90., "c": 101., "v": 2.} for i in range(2)]
    return {"source": "hyperliquid_mainnet_perpetuals", "fetched_at": "1970-01-01T02:00:00+00:00",
            "interval_ms": H, "range": {"start_ms": 0, "end_ms_exclusive": 2 * H},
            "assets": {coin: {"sz_decimals": 5, "candles": deepcopy(rows),
                               "funding": [{"t": i * H + 9, "rate": .00001} for i in range(2)]} for coin in ASSETS}}


class DataSourceTests(unittest.TestCase):
    def test_valid_and_ms_funding_timestamp(self):
        self.assertTrue(validate_dataset(fixture(), 2 * H)["passed"])

    def test_missing_duplicate_future_misaligned_and_bad_ohlc_fail(self):
        changes = [lambda d: d["assets"]["BTC"]["candles"].pop(),
                   lambda d: d["assets"]["BTC"]["candles"].append(d["assets"]["BTC"]["candles"][0]),
                   lambda d: d["assets"]["ETH"]["candles"][0].update(t=1),
                   lambda d: d["assets"]["ETH"]["candles"][0].update(h=99),
                   lambda d: d["assets"]["ETH"]["candles"][0].update(v=float("nan")),
                   lambda d: d["assets"]["ETH"]["funding"].pop(),
                   lambda d: d["assets"]["ETH"]["funding"][0].update(t=H + 1)]
        for change in changes:
            with self.subTest(change=change):
                data = fixture()
                change(data)
                with self.assertRaises(DataIntegrityError):
                    validate_dataset(data, 2 * H)
        with self.assertRaisesRegex(DataIntegrityError, "current or future"):
            validate_dataset(fixture(), 2 * H - 1)

    def test_incomplete_candle_excluded_and_duplicate_rejected(self):
        raw = [{"t": t, "T": t + H - 1, "s": "BTC", "i": "1h",
                "o": "100", "h": "110", "l": "90", "c": "101", "v": "2"} for t in (0, H)]
        rows, exclusions = normalize_candles(raw, "BTC", 0, H)
        self.assertEqual(len(rows), 1)
        self.assertEqual(exclusions["incomplete_or_after_end"], 1)
        with self.assertRaisesRegex(DataIntegrityError, "duplicate"):
            normalize_candles([raw[0], raw[0]], "BTC", 0, H)

    def test_funding_short_pages_are_not_assumed_complete(self):
        class CappedClient:
            def __init__(self): self.calls = []
            def post(self, payload):
                self.calls.append(payload)
                rows = [{"coin": "BTC", "time": i * H + 5, "fundingRate": "0.00001"} for i in range(5)]
                return [r for r in rows if payload["startTime"] <= r["time"] <= payload["endTime"]][:2]
        client = CappedClient()
        rows = fetch_funding(client, "BTC", 0, 5 * H, chunk_hours=3)
        self.assertEqual([r["t"] for r in rows], [i * H + 5 for i in range(5)])
        self.assertEqual(client.calls[1]["startTime"], H + 6)
        self.assertEqual(len(client.calls), 5)

    def test_pagination_out_of_range_or_duplicate_fails(self):
        class BrokenClient:
            def post(self, payload):
                return [{"coin": "BTC", "time": 1, "fundingRate": 0} for _ in range(2)]
        with self.assertRaises(DataIntegrityError):
            fetch_funding(BrokenClient(), "BTC", 0, H)

    def test_retry_timeout_and_allowlist(self):
        attempts, waits = [], []
        def opener(request, timeout):
            attempts.append(timeout)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, "rate limit", {}, None)
            return io.BytesIO(b'{"universe": []}')
        client = PublicInfoClient(opener=opener, sleeper=waits.append, timeout=7)
        self.assertEqual(client.post({"type": "meta"}), {"universe": []})
        self.assertEqual(attempts, [7, 7])
        self.assertEqual(waits, [1])
        for payload in ({"type": "order"}, {"type": "clearinghouseState", "user": "anything"},
                        {"type": "meta", "user": "anything"}):
            with self.assertRaises(DataIntegrityError): client.post(payload)

    def test_sha256_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.json"
            save_dataset(fixture(), path)
            self.assertEqual(load_verified(path)["source"], fixture()["source"])
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(DataIntegrityError, "SHA256"):
                load_verified(path)


if __name__ == "__main__":
    unittest.main()
