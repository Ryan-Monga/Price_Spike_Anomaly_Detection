"""
Evaluation and Metrics Module.
Calculates Precision, Recall, F1, F4 (beta=4), False Positive Rate, Confusion Matrix,
and Detection Delay for synthetic anomaly benchmark scenarios.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

@dataclass
class EvaluationMetrics:
    model_name: str
    precision: float
    recall: float
    f1_score: float
    f4_score: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    detection_delay_mean: Optional[float]
    zero_division_convention: str = "Metrics with zero denominator evaluate safely to 0.0"
    disclaimer: str = (
        "EVALUATION NOTICE: Metrics measure detection fidelity against controlled synthetic "
        "benchmark scenarios only. They do NOT establish legal or real-world accuracy on unverified market schemes."
    )

def calculate_f_beta(precision: float, recall: float, beta: float = 4.0) -> float:
    """
    Calculates generalized F-beta score:
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    For beta = 4, recall is weighted 4 times as heavily as precision.
    Safely returns 0.0 if denominator is 0.
    """
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom <= 1e-9:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)

def compute_detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_windows: Optional[List[Tuple[int, int]]] = None
) -> Optional[float]:
    """
    Calculates average detection delay (in time-steps) from the start of an anomaly window
    to the first predicted positive observation within that window.
    """
    if event_windows is None or len(event_windows) == 0:
        # Infer contiguous positive segments from y_true
        windows = []
        in_window = False
        start = 0
        for i, val in enumerate(y_true):
            if val == 1 and not in_window:
                in_window = True
                start = i
            elif val == 0 and in_window:
                in_window = False
                windows.append((start, i - 1))
        if in_window:
            windows.append((start, len(y_true) - 1))
        event_windows = windows

    if not event_windows:
        return None

    delays = []
    for (start, end) in event_windows:
        sub_pred = y_pred[start : end + 1]
        hits = np.where(sub_pred == 1)[0]
        if len(hits) > 0:
            delays.append(float(hits[0]))  # 0 delay means detected on the first observation
        else:
            # Missed anomaly window
            delays.append(float(end - start + 1))

    return round(float(np.mean(delays)), 2) if delays else None

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Hybrid Detector",
    event_windows: Optional[List[Tuple[int, int]]] = None
) -> EvaluationMetrics:
    """
    Calculates full benchmark surveillance metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Safely compute standard metrics
    prec = precision_score(y_t, y_p, zero_division=0.0)
    rec = recall_score(y_t, y_p, zero_division=0.0)
    f1 = f1_score(y_t, y_p, zero_division=0.0)
    f4 = calculate_f_beta(prec, rec, beta=4.0)

    # False Positive Rate = FP / (FP + TN)
    denom_fpr = fp + tn
    fpr = float(fp / denom_fpr) if denom_fpr > 0 else 0.0

    delay = compute_detection_delay(y_t, y_p, event_windows)

    return EvaluationMetrics(
        model_name=model_name,
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1_score=round(float(f1), 4),
        f4_score=round(float(f4), 4),
        false_positive_rate=round(float(fpr), 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        total_samples=len(y_t),
        detection_delay_mean=delay
    )

def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    event_windows: Optional[List[Tuple[int, int]]] = None
) -> pd.DataFrame:
    """
    Evaluates multiple models against ground truth and returns a comparative DataFrame.
    """
    records = []
    for model_name, y_pred in predictions_dict.items():
        metrics = evaluate_predictions(y_true, y_pred, model_name=model_name, event_windows=event_windows)
        records.append({
            "Model": metrics.model_name,
            "Precision": metrics.precision,
            "Recall": metrics.recall,
            "F1-Score": metrics.f1_score,
            "F4-Score (Recall Prioritized)": metrics.f4_score,
            "False Positive Rate": metrics.false_positive_rate,
            "True Positives (TP)": metrics.true_positives,
            "False Positives (FP)": metrics.false_positives,
            "True Negatives (TN)": metrics.true_negatives,
            "False Negatives (FN)": metrics.false_negatives,
            "Avg Detection Delay (days)": metrics.detection_delay_mean,
        })
    return pd.DataFrame(records)
