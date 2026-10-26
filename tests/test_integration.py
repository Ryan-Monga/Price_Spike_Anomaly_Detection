"""
End-to-End Integration Test for Price Spike Anomaly Detection Pipeline.
Validates the entire surveillance workflow from offline data generation to model training,
score calibration, anomaly injection, benchmark evaluation, and persistence.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import OneClassSVMConfig, LSTMAutoencoderConfig
from src.model_manager import ModelManager

def test_full_pipeline_offline(tmp_path: Path):
    """
    Complete end-to-end integration test executed without internet access.
    """
    manager = ModelManager(models_dir=tmp_path / "models")

    # 1. Ingest offline dataset
    features_df, meta = manager.load_dataset(ticker="BTC-USD", period="6mo", force_refresh=False, offline_mode=True)
    assert features_df is not None
    assert len(features_df) >= 60
    assert manager.train_df is not None
    assert manager.val_df is not None
    assert manager.test_df is not None

    # 2. Check model status before training
    status_initial = manager.get_model_status()
    assert status_initial["baseline"]["ready"] == True
    assert status_initial["one_class_svm"]["ready"] == False
    assert status_initial["lstm_autoencoder"]["ready"] == False

    # 3. Train One-Class SVM on training split
    svm_meta = manager.train_one_class_svm(OneClassSVMConfig(nu=0.05))
    assert manager.svm_detector.is_trained
    assert "train_observations" in svm_meta

    # 4. Train LSTM Autoencoder on training split
    lstm_cfg = LSTMAutoencoderConfig(
        sequence_length=15,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
        epochs=2,
        batch_size=8,
        seed=42
    )
    lstm_meta = manager.train_lstm_autoencoder(config=lstm_cfg)
    assert manager.lstm_detector.is_trained
    assert len(manager.lstm_detector.training_history["train_loss"]) == 2

    # 5. Calibrate hybrid threshold on validation split
    calibrated_thresh = manager.calibrate_threshold(percentile=95.0)
    assert 0.0 <= calibrated_thresh <= 1.0

    # 6. Run hybrid surveillance prediction
    detected_df = manager.predict_surveillance(manager.test_df)
    assert "hybrid_score" in detected_df.columns
    assert "stat_score" in detected_df.columns
    assert "svm_score" in detected_df.columns
    assert "lstm_score" in detected_df.columns
    assert "detection_status" in detected_df.columns
    assert "alert_explanation" in detected_df.columns

    # 7. Run synthetic anomaly experiment on test split
    mod_test_df, comp_df, record, metrics = manager.run_synthetic_experiment(
        scenario="pump_and_dump_reversal",
        duration=3,
        strength=1.5,
        seed=42
    )
    assert comp_df is not None
    assert len(comp_df) >= 3  # Baseline, Hybrid, SVM, LSTM
    assert "Precision" in comp_df.columns
    assert "Recall" in comp_df.columns
    assert "F4-Score (Recall Prioritized)" in comp_df.columns

    # 8. Check persistence
    pred_path = manager.storage.export_predictions_csv(detected_df, ticker="BTC-USD")
    assert pred_path.exists()

    eval_path = manager.storage.export_experiment_csv(comp_df, prefix="integration_eval")
    assert eval_path.exists()

    past_exps = manager.storage.get_past_experiments(limit=5)
    assert len(past_exps) > 0
