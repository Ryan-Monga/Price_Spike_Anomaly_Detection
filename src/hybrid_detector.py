"""
Hybrid Anomaly Detector Module.
Ensembles Statistical Baseline, One-Class SVM, and LSTM Autoencoder detectors
using calibrated scores, configurable weights, and documented decision thresholds.
"""
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import HybridDetectorConfig
from src.baseline import StatisticalBaselineDetector
from src.one_class_svm import OneClassSVMDetector
from src.lstm_autoencoder import LSTMAutoencoderDetector

logger = logging.getLogger(__name__)

@dataclass
class HybridDetectionResult:
    hybrid_score: float
    threshold: float
    is_anomaly: bool
    status: str  # "ANOMALY" or "NORMAL"
    stat_score: float
    svm_score: Optional[float]
    lstm_score: Optional[float]
    explanation: str

class HybridAnomalyDetector:
    """
    Combines baseline, SVM, and LSTM scores into a weighted hybrid score.
    Supports dynamic fallback when advanced models are not yet trained.
    """
    def __init__(
        self,
        baseline_detector: Optional[StatisticalBaselineDetector] = None,
        svm_detector: Optional[OneClassSVMDetector] = None,
        lstm_detector: Optional[LSTMAutoencoderDetector] = None,
        config: Optional[HybridDetectorConfig] = None
    ):
        self.baseline_detector = baseline_detector or StatisticalBaselineDetector()
        self.svm_detector = svm_detector
        self.lstm_detector = lstm_detector
        self.config = config or HybridDetectorConfig()
        self.threshold = self.config.threshold

    def set_weights(self, w_stat: float, w_svm: float, w_lstm: float) -> None:
        """Sets and normalizes component weights to sum to 1.0."""
        total = w_stat + w_svm + w_lstm
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        self.config.w_stat = w_stat / total
        self.config.w_svm = w_svm / total
        self.config.w_lstm = w_lstm / total

    def set_threshold(self, threshold: float) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def calibrate_threshold_on_validation(self, val_df_features: pd.DataFrame, percentile: float = 95.0) -> float:
        """
        Determines the decision threshold from presumed-normal validation observations.
        Freezes this threshold before test evaluation.
        """
        df_scored = self.detect_dataframe(val_df_features)
        scores = df_scored["hybrid_score"].values
        calibrated_thresh = float(np.percentile(scores, percentile))
        self.threshold = round(calibrated_thresh, 4)
        logger.info(f"Calibrated hybrid threshold to {self.threshold} ({percentile}th percentile of validation)")
        return self.threshold

    def detect_dataframe(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Runs available detectors on df_features and calculates the composite hybrid score.
        
        Args:
            df_features: DataFrame with engineered features.
            
        Returns:
            DataFrame augmented with:
            - stat_score, stat_anomaly, stat_reasons
            - svm_score, svm_anomaly (if available)
            - lstm_score, lstm_anomaly (if available)
            - hybrid_score in [0, 1]
            - hybrid_threshold
            - is_anomaly: bool
            - detection_status: "ANOMALY" or "NORMAL"
            - alert_explanation: detailed interpretable description
        """
        df_out = df_features.copy()

        # 1. Statistical Baseline (always available)
        df_out = self.baseline_detector.detect(df_out)
        stat_scores = df_out["stat_score"].values

        # 2. One-Class SVM (if trained)
        has_svm = self.svm_detector is not None and getattr(self.svm_detector, "is_trained", False)
        if has_svm:
            df_out = self.svm_detector.transform_dataframe(df_out)
            svm_scores = df_out["svm_score"].values
        else:
            svm_scores = None
            df_out["svm_score"] = np.nan
            df_out["svm_anomaly"] = False

        # 3. LSTM Autoencoder (if trained)
        has_lstm = self.lstm_detector is not None and getattr(self.lstm_detector, "is_trained", False)
        if has_lstm:
            df_out = self.lstm_detector.transform_dataframe(df_out)
            lstm_scores = df_out["lstm_score"].values
        else:
            lstm_scores = None
            df_out["lstm_score"] = np.nan
            df_out["lstm_anomaly"] = False

        # 4. Normalize weights based on active models
        active_weights = {}
        active_weights["stat"] = self.config.w_stat
        if has_svm:
            active_weights["svm"] = self.config.w_svm
        if has_lstm:
            active_weights["lstm"] = self.config.w_lstm

        sum_w = sum(active_weights.values())
        norm_w = {k: v / sum_w for k, v in active_weights.items()}

        # 5. Compute hybrid weighted score with per-row valid model reweighting
        n = len(df_out)
        hybrid_scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            row_w = {}
            row_s = {}
            if not np.isnan(stat_scores[i]):
                row_w["stat"] = self.config.w_stat
                row_s["stat"] = stat_scores[i]
            if has_svm and not np.isnan(svm_scores[i]):
                row_w["svm"] = self.config.w_svm
                row_s["svm"] = svm_scores[i]
            if has_lstm and not np.isnan(lstm_scores[i]):
                row_w["lstm"] = self.config.w_lstm
                row_s["lstm"] = lstm_scores[i]

            sum_row = sum(row_w.values())
            if sum_row > 0:
                score = sum((w / sum_row) * row_s[k] for k, w in row_w.items())
            else:
                score = stat_scores[i]
            hybrid_scores[i] = float(np.clip(score, 0.0, 1.0))

        df_out["hybrid_score"] = hybrid_scores
        df_out["hybrid_threshold"] = self.threshold

        is_anomaly = hybrid_scores >= self.threshold
        df_out["is_anomaly"] = is_anomaly
        df_out["detection_status"] = np.where(is_anomaly, "ANOMALY", "NORMAL")

        # 6. Synthesize alert explanations
        explanations: List[str] = []
        for i in range(n):
            exp_parts = []
            flagged = is_anomaly[i]

            # Score summary
            exp_parts.append(
                f"Hybrid risk score {hybrid_scores[i]:.3f} vs threshold {self.threshold:.3f}."
            )

            # Component details if anomalous or notable
            if flagged or hybrid_scores[i] >= 0.45:
                stat_r = df_out["stat_reasons"].iloc[i]
                if stat_r and "Normal market" not in stat_r:
                    exp_parts.append(f"Statistical indicator: {stat_r}")

                if has_svm and df_out["svm_anomaly"].iloc[i]:
                    exp_parts.append(
                        f"One-Class SVM flagged feature vector as outside normal training boundary (SVM score: {svm_scores[i]:.3f})."
                    )

                if has_lstm and df_out["lstm_anomaly"].iloc[i]:
                    exp_parts.append(
                        f"LSTM Autoencoder detected high temporal reconstruction error (LSTM score: {lstm_scores[i]:.3f}), indicating an abnormal price-volume sequence pattern."
                    )
            
            if not flagged and len(exp_parts) == 1:
                exp_parts.append("All technical, distribution, and temporal sequence indicators remain within normal expected tolerance bounds.")

            explanations.append(" | ".join(exp_parts))

        df_out["alert_explanation"] = explanations
        return df_out
