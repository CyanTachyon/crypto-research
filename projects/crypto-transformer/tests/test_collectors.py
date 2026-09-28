import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from crypto_transformer.data.collectors.binance import fetch_ohlcv, _ohlcv_to_df, COLUMNS


class TestOhlcvToDf:
    def test_basic_conversion(self):
        rows = [
            [1700000000000, 100.0, 105.0, 98.0, 102.0, 500.0],
            [1700003600000, 102.0, 108.0, 100.0, 106.0, 600.0],
        ]
        df = _ohlcv_to_df(rows)
        assert list(df.columns) == COLUMNS
        assert len(df) == 2
        assert df["close"].dtype == float

    def test_deduplication(self):
        rows = [
            [1700000000000, 100.0, 105.0, 98.0, 102.0, 500.0],
            [1700000000000, 100.0, 105.0, 98.0, 102.0, 500.0],
            [1700003600000, 102.0, 108.0, 100.0, 106.0, 600.0],
        ]
        df = _ohlcv_to_df(rows)
        assert len(df) == 2

    def test_sorted_by_timestamp(self):
        rows = [
            [1700003600000, 102.0, 108.0, 100.0, 106.0, 600.0],
            [1700000000000, 100.0, 105.0, 98.0, 102.0, 500.0],
        ]
        df = _ohlcv_to_df(rows)
        assert df["timestamp"].iloc[0] < df["timestamp"].iloc[1]

    def test_empty_rows(self):
        df = _ohlcv_to_df([])
        assert len(df) == 0
        assert list(df.columns) == COLUMNS


class TestBinanceIntegration:
    def test_fetch_returns_dataframe(self):
        df = fetch_ohlcv("ETH/USDT", "1h", start_date="2025-06-01", use_cache=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "close" in df.columns
        assert "timestamp" in df.columns
