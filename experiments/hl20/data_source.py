"""Read-only public market data for the BTC/ETH, 20-USDC experiment.

No credentials, account endpoints, exchange clients or order methods are used.
Funding timestamps are actual settlement timestamps (often a few milliseconds
after the hour); rates are per hourly settlement, not annualized or divided by 8.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request

HOUR_MS = 3_600_000
ASSETS = ("BTC", "ETH")
INFO_URL = "https://api.hyperliquid.xyz/info"
DOCS = [
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#candle-snapshot",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding",
]


def utc_iso(ms: int | None = None) -> str:
    return datetime.fromtimestamp((ms if ms is not None else int(time.time() * 1000)) / 1000,
                                  timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DataIntegrityError(ValueError):
    """Fail closed on invalid, incomplete or misaligned source data."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataIntegrityError(message)


class PublicInfoClient:
    """Fixed /info endpoint and explicit market-data allowlist, with bounded retries."""

    def __init__(self, raw_dir: Path | None = None, timeout: float = 25,
                 attempts: int = 5, opener: Callable = urllib.request.urlopen,
                 sleeper: Callable = time.sleep):
        self.raw_dir, self.timeout, self.attempts = raw_dir, timeout, attempts
        self.opener, self.sleeper = opener, sleeper
        self.requests: list[dict] = []
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)

    def post(self, payload: dict) -> Any:
        kind = payload.get("type")
        allowed = {"meta": {"type"}, "candleSnapshot": {"type", "req"},
                   "fundingHistory": {"type", "coin", "startTime", "endTime"}}
        require(kind in allowed and set(payload) == allowed[kind], "Not an allowed public market-data request")
        if kind == "candleSnapshot":
            require(set(payload["req"]) == {"coin", "interval", "startTime", "endTime"}, "Invalid candle request")
            require(payload["req"]["coin"] in ASSETS and payload["req"]["interval"] == "1h", "Only BTC/ETH 1h is supported")
        if kind == "fundingHistory":
            require(payload["coin"] in ASSETS, "Only BTC/ETH is supported")
        request = urllib.request.Request(INFO_URL, data=canonical_bytes(payload),
                                         headers={"Content-Type": "application/json", "User-Agent": "hl20-readonly-research/1"})
        for attempt in range(self.attempts):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    body = response.read()
                value = json.loads(body)
                record = {"payload": payload, "fetched_at": utc_iso(), "response_sha256": sha256(body),
                          "response_bytes": len(body), "attempts": attempt + 1}
                if self.raw_dir:
                    name = f"{len(self.requests):04d}_{kind}.json"
                    (self.raw_dir / name).write_bytes(body)
                    record["raw_file"] = "raw/" + name
                self.requests.append(record)
                return value
            except urllib.error.HTTPError as error:
                if error.code != 429 and not 500 <= error.code <= 599:
                    raise
                if attempt + 1 == self.attempts:
                    raise
                self.sleeper(min(2 ** attempt, 16))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt + 1 == self.attempts:
                    raise
                self.sleeper(min(2 ** attempt, 16))
        raise RuntimeError("Unreachable retry exhaustion")


def numeric(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataIntegrityError(f"Invalid numeric {label}") from error
    require(math.isfinite(result), f"Non-finite {label}")
    return result


def normalize_candles(raw: list, coin: str, start: int, end: int) -> tuple[list, dict]:
    require(isinstance(raw, list), f"{coin}: candle response is not a list")
    candles, excluded = [], {"before_requested_start": 0, "incomplete_or_after_end": 0}
    seen = set()
    for row in raw:
        t = row["t"]
        require(isinstance(t, int) and t % HOUR_MS == 0, f"{coin}: non-hourly candle timestamp")
        require(t not in seen, f"{coin}: duplicate candle timestamp {t}")
        seen.add(t)
        require(row.get("s") == coin and row.get("i") == "1h", f"{coin}: incorrect asset/interval")
        require(row.get("T") == t + HOUR_MS - 1, f"{coin}: incorrect candle close timestamp")
        if t < start:
            excluded["before_requested_start"] += 1
            continue
        if t + HOUR_MS > end:
            excluded["incomplete_or_after_end"] += 1
            continue
        candles.append({"t": t, **{field: numeric(row[field], f"{coin}/{field}") for field in "ohlcv"}})
    candles.sort(key=lambda r: r["t"])
    return candles, excluded


def fetch_funding(client: PublicInfoClient, coin: str, start: int, end: int,
                  chunk_hours: int = 500) -> list[dict]:
    """Window and paginate inclusive endpoints; never infer completeness from page size.

    A short page might still be truncated. Continue at last timestamp + 1 until
    an empty response or the requested end; integrity validation catches gaps.
    """
    require(end > start and chunk_hours > 0, "Invalid funding range")
    records: list[dict] = []
    seen: set[int] = set()
    for window_start in range(start, end, chunk_hours * HOUR_MS):
        window_end = min(end, window_start + chunk_hours * HOUR_MS)
        cursor = window_start
        while cursor < window_end:
            page = client.post({"type": "fundingHistory", "coin": coin,
                                "startTime": cursor, "endTime": window_end - 1})
            require(isinstance(page, list), f"{coin}: funding response is not a list")
            if not page:
                break
            latest = cursor - 1
            for row in page:
                t = row["time"]
                require(isinstance(t, int) and cursor <= t < window_end, f"{coin}: funding page outside request")
                require(row.get("coin") == coin, f"{coin}: wrong funding asset")
                require(t not in seen, f"{coin}: duplicate funding {t}")
                seen.add(t)
                latest = max(latest, t)
                records.append({"t": t, "rate": numeric(row["fundingRate"], f"{coin}/fundingRate")})
            require(latest >= cursor, f"{coin}: funding pagination made no progress")
            cursor = latest + 1
    return sorted(records, key=lambda r: r["t"])


def validate_dataset(dataset: dict, now_ms: int | None = None) -> dict:
    require(dataset.get("source") == "hyperliquid_mainnet_perpetuals", "Expected actual Hyperliquid perpetuals dataset")
    require(dataset.get("interval_ms") == HOUR_MS, "Expected hourly data")
    require(set(dataset["assets"]) == set(ASSETS), "Expected exactly BTC and ETH")
    start, end = dataset["range"]["start_ms"], dataset["range"]["end_ms_exclusive"]
    require(isinstance(start, int) and isinstance(end, int) and 0 <= start < end, "Invalid dataset range")
    require(start % HOUR_MS == 0 and end % HOUR_MS == 0, "Range must be hour-aligned")
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    require(end <= now_ms // HOUR_MS * HOUR_MS, "Dataset contains current or future candles")
    expected = list(range(start, end, HOUR_MS))
    require(len(expected) <= 5000, "API only retains the most recent 5000 candles")
    for coin in ASSETS:
        asset = dataset["assets"][coin]
        require(type(asset["sz_decimals"]) is int and 0 <= asset["sz_decimals"] <= 8, f"{coin}: invalid size precision")
        candles, funding = asset["candles"], asset["funding"]
        require([r["t"] for r in candles] == expected, f"{coin}: candle gaps, duplicates, ordering or alignment error")
        for row in candles:
            vals = {field: numeric(row[field], f"{coin}/{field}") for field in "ohlcv"}
            require(min(vals[k] for k in "ohlc") > 0 and vals["v"] >= 0, f"{coin}: invalid price/volume")
            require(vals["l"] <= min(vals["o"], vals["c"]) <= max(vals["o"], vals["c"]) <= vals["h"], f"{coin}: impossible OHLC")
        require(all(isinstance(r["t"], int) and start <= r["t"] < end for r in funding), f"{coin}: invalid funding timestamp")
        require([r["t"] // HOUR_MS * HOUR_MS for r in funding] == expected,
                f"{coin}: missing, duplicated or unordered hourly funding settlements")
        for row in funding:
            numeric(row["rate"], f"{coin}/funding")
    return {"passed": True, "candles_per_asset": len(expected), "hourly_funding_per_asset": len(expected),
            "start_utc": utc_iso(start), "end_exclusive_utc": utc_iso(end),
            "checks": ["hourly_continuity", "cross_asset_alignment", "completed_candles_only",
                       "no_duplicates", "finite_prices_volume_funding", "ohlc_bounds", "hourly_funding_coverage"]}


def collect_dataset(client: PublicInfoClient, hours: int = 5000, now_ms: int | None = None) -> dict:
    require(1 <= hours <= 5000, "hours must be between 1 and 5000")
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    end = now_ms // HOUR_MS * HOUR_MS
    requested_start = end - hours * HOUR_MS
    meta = client.post({"type": "meta"})
    require(isinstance(meta, dict) and isinstance(meta.get("universe"), list), "Invalid perpetual metadata")
    universe = {row["name"]: row for row in meta["universe"]}
    assets, exclusions = {}, {}
    for coin in ASSETS:
        require(coin in universe and not universe[coin].get("isDelisted", False), f"{coin}: unavailable perpetual market")
        raw = client.post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1h",
                                                               "startTime": requested_start, "endTime": end - 1}})
        candles, exclusions[coin] = normalize_candles(raw, coin, requested_start, end)
        require(bool(candles), f"{coin}: no complete candles returned")
        # Retention may include the in-progress candle. Only one unavailable
        # oldest hour is allowed, never a silent truncation to an arbitrary range.
        require(candles[0]["t"] <= requested_start + (HOUR_MS if hours == 5000 else 0), f"{coin}: unexpected retention truncation")
        assets[coin] = {"sz_decimals": universe[coin]["szDecimals"], "candles": candles,
                        "metadata_at_collection": universe[coin]}
    start = max(asset["candles"][0]["t"] for asset in assets.values())
    for coin, asset in assets.items():
        exclusions[coin]["alignment_prefix"] = sum(r["t"] < start for r in asset["candles"])
        asset["candles"] = [r for r in asset["candles"] if r["t"] >= start]
        asset["funding"] = fetch_funding(client, coin, start, end)
    dataset = {"schema_version": 1, "source": "hyperliquid_mainnet_perpetuals", "fetched_at": utc_iso(),
               "interval_ms": HOUR_MS, "range": {"start_ms": start, "end_ms_exclusive": end},
               "assets": assets, "provenance": {"endpoint": INFO_URL, "official_docs": DOCS,
               "requested_hours": hours, "requested_start_ms": requested_start, "cutoff_now_ms": now_ms,
               "funding_semantics": "Actual hourly settled funding rates; t preserves API settlement milliseconds. Positive rate: longs pay shorts.",
               "limitations": ["Most recent 5000 candles only; at most one oldest hour removed for retention/alignment.",
                               "OHLC are trade prices, not historical mark/oracle prices; funding cash amounts need an approximation if oracle history is absent.",
                               "Present size precision does not prove historical precision was identical."],
               "excluded_candles": exclusions, "requests": client.requests}}
    dataset["integrity"] = validate_dataset(dataset, now_ms)
    return dataset


def save_dataset(dataset: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(dataset)
    path.write_bytes(body)
    manifest = {"filename": path.name, "sha256": sha256(body), "bytes": len(body),
                "source": dataset["source"], "fetched_at": dataset["fetched_at"],
                "integrity": dataset.get("integrity"), "range": dataset.get("range")}
    path.with_suffix(".manifest.json").write_bytes(canonical_bytes(manifest))
    return manifest


def load_verified(path: Path) -> dict:
    raw = path.read_bytes()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    require(sha256(raw) == manifest["sha256"], "Dataset SHA256 mismatch")
    dataset = json.loads(raw)
    validate_dataset(dataset)
    return dataset


def export_binance_reference(root: Path, out: Path) -> dict:
    """Export old caches without interpolating aggregated funding into fake events.

    pandas is optional and imported only for this research reference export.
    Native Binance candles are an external-venue reference, never HL validation.
    """
    import pandas as pd
    assets, sources = {}, []
    for coin in ASSETS:
        candle_path = root / "data" / "raw" / f"binance_{coin}_USDT_1h.parquet"
        funding_path = root / "data" / "external" / f"funding_rate_{coin}USDT.parquet"
        candles_df, funding_df = pd.read_parquet(candle_path), pd.read_parquet(funding_path)
        for path in (candle_path, funding_path):
            sources.append({"path": str(path), "sha256": sha256(path.read_bytes())})
        require("timestamp" in candles_df.columns, "Expected timestamp column in candle cache")
        times = pd.to_datetime(candles_df["timestamp"], utc=True)
        candles = [{"t": int(timestamp.value // 1_000_000),
                    **{short: numeric(row[long], f"{coin}/{long}") for short, long in
                       zip("ohlcv", ("open", "high", "low", "close", "volume"))}}
                   for timestamp, (_, row) in zip(times, candles_df.iterrows())]
        candles.sort(key=lambda r: r["t"])
        ts = [r["t"] for r in candles]
        require(len(ts) == len(set(ts)), f"{coin}: duplicate cached candle timestamps")
        gaps = [{"after_ms": a, "before_ms": b, "missing_hours": (b - a) // HOUR_MS - 1}
                for a, b in zip(ts, ts[1:]) if b - a != HOUR_MS]
        # Preserve original column names and aggregation exactly; do not fabricate
        # a settlement rate/time from daily mean/max/min/count cache columns.
        funding_records = json.loads(funding_df.to_json(orient="records", date_format="iso"))
        assets[coin] = {"candles": candles, "funding_daily_aggregates": funding_records,
                        "funding_columns": list(funding_df.columns), "candle_gap_report": gaps,
                        "range": {"start_ms": ts[0], "end_ms_exclusive": ts[-1] + HOUR_MS}}
    dataset = {"schema_version": 1, "source": "binance_spot_ohlcv_with_binance_futures_daily_aggregated_funding_cache",
               "fetched_at": utc_iso(), "interval_ms": HOUR_MS, "assets": assets,
               "provenance": {"files": sources, "funding_aggregated": True,
                              "limitations": ["Existing Binance spot OHLCV caches; not Hyperliquid perpetual trade prices.",
                                              "Funding cache is daily aggregated Binance futures data, not hourly or event-level settlements.",
                                              "Do not use these daily summaries as event-level funding or confirmatory Hyperliquid validation.",
                                              "Existing cache candle gaps are reported, never silently filled."]}}
    return save_dataset(dataset, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=5000)
    parser.add_argument("--export-binance-root", type=Path)
    args = parser.parse_args()
    client = PublicInfoClient(raw_dir=args.out_dir / "raw")
    dataset = collect_dataset(client, args.hours)
    print(json.dumps(save_dataset(dataset, args.out_dir / "hyperliquid_1h.json"), indent=2), flush=True)
    if args.export_binance_root:
        print(json.dumps(export_binance_reference(args.export_binance_root, args.out_dir / "binance_reference_1h.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
