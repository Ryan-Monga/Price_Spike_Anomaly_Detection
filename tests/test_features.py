"""
Tests for Feature Engineering Module.
Verifies return, volatility, z-score, relative volume, zero-division safety,
and causal consistency.
"""
import pytest
import numpy as np
import pandas as pd

from config import CORE_FEATURE_COLUMNS
from src.features import compute_market_features, get_feature_matrix

def create_sample_price_data(n: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range(start="2025-01-01", periods=n)
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, size=n)))
    volume = 1000000 * np.exp(np.random.normal(0, 0.3, size=n))
    return pd.DataFrame({
        "Open": close * 0.99,
        "High": close * 1.02,
        "Low": close * 0.98,
        "Close": close,
        "Adj Close": close,
        "Volume": volume
    }, index=dates)

def test_feature_columns_and_non_null():
    df = create_sample_price_data(60)
    feat_df = compute_market_features(df, rolling_window=20, min_periods=5)
    
    for col in CORE_FEATURE_COLUMNS:
        assert col in feat_df.columns
        assert not feat_df[col].isnull().any()
        assert not np.isinf(feat_df[col]).any()

def test_causal_integrity():
    """Future price manipulation must NOT affect past feature values."""
    df1 = create_sample_price_data(60)
    df2 = df1.copy()
    
    # Modify future observations (rows 50 to 59)
    df2.iloc[50:, df2.columns.get_loc("Close")] *= 2.5
    df2.iloc[50:, df2.columns.get_loc("Volume")] *= 5.0

    feat1 = compute_market_features(df1, rolling_window=20)
    feat2 = compute_market_features(df2, rolling_window=20)

    # First 50 rows must remain mathematically IDENTICAL
    for col in CORE_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            feat1[col].iloc[:50].values,
            feat2[col].iloc[:50].values,
            rtol=1e-5,
            err_msg=f"Lookahead leakage detected in column {col}"
        )

def test_zero_and_infinite_division_safety():
    df = create_sample_price_data(40)
    # Inject zero volume and flat prices
    df.loc[df.index[10], "Volume"] = 0.0
    df.loc[df.index[11], "Volume"] = 0.0
    df.loc[df.index[15], "Close"] = df.loc[df.index[14], "Close"]

    feat_df = compute_market_features(df)
    for col in CORE_FEATURE_COLUMNS:
        assert not feat_df[col].isnull().any()
        assert not np.isinf(feat_df[col]).any()
