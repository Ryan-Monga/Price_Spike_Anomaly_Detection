"""
Tests for Preprocessing and Chronological Splitting Module.
Verifies no-shuffle train/val/test splits, scaler isolation, and causal sequence construction.
"""
import pytest
import numpy as np
import pandas as pd

from config import CORE_FEATURE_COLUMNS
from src.features import compute_market_features
from src.preprocessing import chronological_split, FeatureScaler, create_causal_sequences

def make_test_features(n: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(42)
    close = 150.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=n)))
    volume = 5000000 * np.exp(np.random.normal(0, 0.3, size=n))
    df = pd.DataFrame({
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": volume
    }, index=dates)
    return compute_market_features(df, rolling_window=20)

def test_chronological_split():
    df = make_test_features(100)
    train_df, val_df, test_df, meta = chronological_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strictly chronological timestamps
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

def test_scaler_isolation():
    df = make_test_features(80)
    train_df, val_df, test_df, _ = chronological_split(df, 0.7, 0.15, 0.15)

    scaler = FeatureScaler(scaler_type="robust")
    assert not scaler.is_fitted

    # Transform without fit must raise error
    with pytest.raises(RuntimeError):
        scaler.transform(test_df)

    scaler.fit(train_df)
    assert scaler.is_fitted

    scaled_train = scaler.transform(train_df)
    scaled_test = scaler.transform(test_df)

    assert scaled_train.shape == (len(train_df), len(CORE_FEATURE_COLUMNS))
    assert scaled_test.shape == (len(test_df), len(CORE_FEATURE_COLUMNS))

def test_causal_sequence_generation():
    df = make_test_features(50)
    scaler = FeatureScaler().fit(df)
    scaled = scaler.transform(df)
    seq_len = 20

    seqs, target_times = create_causal_sequences(scaled, df.index, sequence_length=seq_len)
    
    expected_count = 50 - seq_len + 1
    assert seqs.shape == (expected_count, seq_len, len(CORE_FEATURE_COLUMNS))
    assert len(target_times) == expected_count

    # The target timestamp of the first sequence must match the seq_len-th observation (index seq_len - 1)
    assert target_times[0] == df.index[seq_len - 1]
    assert target_times[-1] == df.index[-1]
