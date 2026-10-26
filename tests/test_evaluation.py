"""
Tests for Evaluation and Metrics Module.
Verifies Precision, Recall, F1, F4 (beta=4), False Positive Rate,
Confusion Matrix, and Detection Delay calculation.
"""
import pytest
import numpy as np
import pandas as pd

from src.evaluation import calculate_f_beta, compute_detection_delay, evaluate_predictions, compare_models

def test_f4_calculation():
    # If precision=0.5, recall=1.0:
    # beta=4, beta^2=16
    # F4 = 17 * (0.5 * 1.0) / (16 * 0.5 + 1.0) = 8.5 / (8.0 + 1.0) = 8.5 / 9.0 = 0.9444...
    f4 = calculate_f_beta(precision=0.5, recall=1.0, beta=4.0)
    assert pytest.approx(f4, abs=1e-3) == 0.9444

    # If precision=1.0, recall=0.5:
    # F4 = 17 * (1.0 * 0.5) / (16 * 1.0 + 0.5) = 8.5 / 16.5 = 0.5151...
    # Shows F4 prioritizes recall heavily!
    f4_low_recall = calculate_f_beta(precision=1.0, recall=0.5, beta=4.0)
    assert pytest.approx(f4_low_recall, abs=1e-3) == 0.5151
    assert f4 > f4_low_recall

def test_zero_division_safety():
    assert calculate_f_beta(0.0, 0.0, beta=4.0) == 0.0
    
    # All zeros
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.f4_score == 0.0
    assert metrics.false_positive_rate == 0.0

def test_evaluate_predictions_metrics():
    # 2 true positives, 1 false positive, 1 false negative, 6 true negatives
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    metrics = evaluate_predictions(y_true, y_pred, model_name="TestModel")
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 6

    # Precision = 2 / 3 = 0.6667
    assert pytest.approx(metrics.precision, abs=1e-3) == 0.6667
    # Recall = 2 / 3 = 0.6667
    assert pytest.approx(metrics.recall, abs=1e-3) == 0.6667
    # FPR = FP / (FP + TN) = 1 / 7 = 0.1429
    assert pytest.approx(metrics.false_positive_rate, abs=1e-3) == 0.1429

def test_detection_delay():
    # Anomaly window from index 2 to index 5
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    # Detector flags at index 3 (delay = 1 step)
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0])

    delay = compute_detection_delay(y_true, y_pred)
    assert delay == 1.0

def test_compare_models_table():
    y_true = np.array([0, 0, 1, 1, 0, 0])
    pred_dict = {
        "Baseline": np.array([0, 0, 1, 0, 0, 0]),
        "Hybrid": np.array([0, 0, 1, 1, 0, 0])
    }
    df_comp = compare_models(y_true, pred_dict)
    assert len(df_comp) == 2
    assert "Precision" in df_comp.columns
    assert "F4-Score (Recall Prioritized)" in df_comp.columns
