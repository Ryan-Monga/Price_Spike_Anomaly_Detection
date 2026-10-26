"""
Tests for Statistical Baseline Anomaly Detector.
Verifies rule evaluation, continuous scores in [0, 1], and reason formatting.
"""
import pytest
import numpy as np
import pandas as pd

from config import BaselineConfig
from src.features import compute_market_features
from src.baseline import StatisticalBaselineDetector

def make_sample_data(n: int = 50) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, size=n)))
    volume = 1000000 * np.exp(np.random.normal(0, 0.2, size=n))
    df = pd.DataFrame({
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": volume
    }, index=dates)
    return compute_market_features(df, rolling_window=20)

def test_baseline_detection_normal():
    df = make_sample_data(50)
    detector = StatisticalBaselineDetector(BaselineConfig(zscore_threshold=3.0, relative_volume_threshold=2.5))
    df_res = detector.detect(df)

    assert "stat_score" in df_res.columns
    assert "stat_anomaly" in df_res.columns
    assert "stat_reasons" in df_res.columns

    # Check score bounds
    assert (df_res["stat_score"] >= 0.0).all()
    assert (df_res["stat_score"] <= 1.0).all()

def test_baseline_detection_abnormal_surge():
    df = make_sample_data(50)
    # Inject large shock on row 45
    target_date = df.index[45]
    df.loc[target_date, "Close"] *= 1.25
    df.loc[target_date, "Volume"] *= 5.0
    
    # Re-compute features
    df = compute_market_features(df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]], rolling_window=20)

    detector = StatisticalBaselineDetector(BaselineConfig(zscore_threshold=2.5, relative_volume_threshold=2.0))
    df_res = detector.detect(df)

    assert df_res.loc[target_date, "stat_anomaly"] == True
    assert df_res.loc[target_date, "stat_score"] > 0.4
    assert "Positive price surge" in df_res.loc[target_date, "stat_reasons"] or "z-score" in df_res.loc[target_date, "stat_reasons"]
