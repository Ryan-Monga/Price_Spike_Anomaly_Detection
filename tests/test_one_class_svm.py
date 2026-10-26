"""
Tests for One-Class SVM Anomaly Detector.
Verifies fitting on normal data, score orientation (larger = more anomalous),
flags, schema validation, and persistence.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import OneClassSVMConfig, CORE_FEATURE_COLUMNS
from src.features import compute_market_features
from src.one_class_svm import OneClassSVMDetector, ModelIncompatibilityError

def make_sample_data(n: int = 80) -> pd.DataFrame:
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

def test_svm_fit_predict_and_orientation():
    train_df = make_sample_data(80)
    detector = OneClassSVMDetector(OneClassSVMConfig(nu=0.05))
    detector.fit(train_df)

    assert detector.is_trained
    assert len(detector.training_metadata) > 0

    scores, flags = detector.predict_scores(train_df)
    assert len(scores) == len(train_df)
    assert len(flags) == len(train_df)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Create an extreme artificial outlier
    test_df = train_df.copy()
    test_df.iloc[-1, test_df.columns.get_loc("return_zscore")] = 12.0
    test_df.iloc[-1, test_df.columns.get_loc("relative_volume")] = 15.0

    outlier_scores, outlier_flags = detector.predict_scores(test_df)
    # The artificial outlier MUST have a higher anomaly score than normal observations
    assert outlier_scores[-1] > np.median(scores)

def test_svm_schema_validation():
    df = make_sample_data(40)
    detector = OneClassSVMDetector()
    detector.fit(df)

    # Drop a feature
    bad_df = df.drop(columns=["return_zscore"])
    with pytest.raises(ModelIncompatibilityError):
        detector.predict_scores(bad_df)

def test_svm_persistence(tmp_path: Path):
    df = make_sample_data(60)
    detector = OneClassSVMDetector()
    detector.fit(df)

    save_file = tmp_path / "test_svm.joblib"
    detector.save(save_file)
    assert save_file.exists()

    loaded = OneClassSVMDetector.load(save_file)
    assert loaded.is_trained
    assert loaded.config.kernel == detector.config.kernel

    s1, f1 = detector.predict_scores(df)
    s2, f2 = loaded.predict_scores(df)
    np.testing.assert_allclose(s1, s2, atol=1e-5)
    assert (f1 == f2).all()
