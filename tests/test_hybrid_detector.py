"""
Tests for Hybrid Anomaly Detector.
Verifies weighted score combinations, fallback when advanced models are missing,
validation threshold calibration, and alert explanation generation.
"""
import pytest
import numpy as np
import pandas as pd

from config import HybridDetectorConfig
from src.features import compute_market_features
from src.baseline import StatisticalBaselineDetector
from src.hybrid_detector import HybridAnomalyDetector

def make_sample_data(n: int = 50) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=n)))
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

def test_hybrid_fallback_baseline_only():
    df = make_sample_data(50)
    baseline = StatisticalBaselineDetector()
    hybrid = HybridAnomalyDetector(baseline_detector=baseline, svm_detector=None, lstm_detector=None)
    
    df_res = hybrid.detect_dataframe(df)
    assert "hybrid_score" in df_res.columns
    assert "detection_status" in df_res.columns
    assert "alert_explanation" in df_res.columns

    # When only baseline is available, hybrid_score should match baseline stat_score
    np.testing.assert_allclose(df_res["hybrid_score"].values, df_res["stat_score"].values, atol=1e-5)

def test_hybrid_weights_normalization():
    hybrid = HybridAnomalyDetector()
    hybrid.set_weights(w_stat=1.0, w_svm=2.0, w_lstm=1.0)
    assert pytest.approx(hybrid.config.w_stat) == 0.25
    assert pytest.approx(hybrid.config.w_svm) == 0.50
    assert pytest.approx(hybrid.config.w_lstm) == 0.25

def test_hybrid_threshold_decision():
    df = make_sample_data(40)
    hybrid = HybridAnomalyDetector()
    hybrid.set_threshold(0.80)

    df_res = hybrid.detect_dataframe(df)
    # Check decision rule consistency: ANOMALY iff score >= threshold
    for _, row in df_res.iterrows():
        if row["hybrid_score"] >= 0.80:
            assert row["detection_status"] == "ANOMALY"
            assert row["is_anomaly"] == True
        else:
            assert row["detection_status"] == "NORMAL"
            assert row["is_anomaly"] == False
