"""
Tests for Market Data Acquisition Module.
Verifies offline dataset generation, column schema, caching, and error handling.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config import RAW_COLUMNS
from src.data_fetcher import MarketDataFetcher, DataFetchError

def test_offline_dataset_generation():
    fetcher = MarketDataFetcher()
    df = fetcher.generate_offline_dataset(ticker="AAPL", period="6mo")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 50
    assert isinstance(df.index, pd.DatetimeIndex)
    
    for col in RAW_COLUMNS:
        assert col in df.columns
        assert not df[col].isnull().any()
        if col != "Volume":
            assert (df[col] > 0).all()

def test_fetch_with_offline_fallback():
    fetcher = MarketDataFetcher()
    df, meta = fetcher.fetch_data(
        ticker="TSLA",
        period="1mo",
        force_refresh=False,
        use_offline_fallback=True,
        offline_mode=True
    )
    assert df is not None
    assert len(df) >= 10
    assert "ticker" in meta
    assert meta["ticker"] == "TSLA"

def test_invalid_ticker_and_period():
    fetcher = MarketDataFetcher()
    with pytest.raises(DataFetchError):
        fetcher.fetch_data(ticker="", period="1y")

    with pytest.raises(DataFetchError):
        fetcher.fetch_data(ticker="AAPL", period="100y")
