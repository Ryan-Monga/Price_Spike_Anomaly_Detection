"""
Tests for PyTorch LSTM Autoencoder Module.
Verifies network construction, causal sequence reconstruction, training,
reconstruction error scoring, and model persistence.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import LSTMAutoencoderConfig, CORE_FEATURE_COLUMNS
from src.features import compute_market_features
from src.lstm_autoencoder import LSTMAutoencoderDetector

def make_sample_data(n: int = 70) -> pd.DataFrame:
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

def test_lstm_autoencoder_training_and_inference():
    df = make_sample_data(70)
    train_df = df.iloc[:50]
    val_df = df.iloc[50:]

    cfg = LSTMAutoencoderConfig(
        sequence_length=15,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
        epochs=3,
        batch_size=8,
        seed=42
    )
    detector = LSTMAutoencoderDetector(config=cfg, device="cpu")
    detector.fit(train_df=train_df, val_df=val_df)

    assert detector.is_trained
    assert len(detector.training_history["train_loss"]) == 3
    assert len(detector.training_history["val_loss"]) == 3
    # Loss should be positive finite
    assert all(l > 0 for l in detector.training_history["train_loss"])

    scores, flags, timestamps = detector.predict_sequences(val_df)
    assert len(scores) == len(val_df) - 15 + 1
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert len(timestamps) == len(scores)

def test_lstm_transform_dataframe():
    df = make_sample_data(60)
    cfg = LSTMAutoencoderConfig(sequence_length=15, epochs=2, batch_size=8)
    detector = LSTMAutoencoderDetector(config=cfg, device="cpu").fit(df)

    df_out = detector.transform_dataframe(df)
    assert "lstm_score" in df_out.columns
    assert "lstm_anomaly" in df_out.columns
    assert len(df_out) == len(df)
    assert not df_out["lstm_score"].isnull().any()

def test_lstm_persistence(tmp_path: Path):
    df = make_sample_data(60)
    cfg = LSTMAutoencoderConfig(sequence_length=15, epochs=2, batch_size=8)
    detector = LSTMAutoencoderDetector(config=cfg, device="cpu").fit(df)

    save_path = tmp_path / "test_lstm.pt"
    detector.save(save_path)
    assert save_path.exists()

    loaded = LSTMAutoencoderDetector.load(save_path, device="cpu")
    assert loaded.is_trained
    assert loaded.config.sequence_length == 15

    scores1, _, _ = detector.predict_sequences(df)
    scores2, _, _ = loaded.predict_sequences(df)
    np.testing.assert_allclose(scores1, scores2, atol=1e-4)
