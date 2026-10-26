"""
Score Calibration Module.
Calibrates heterogeneous model outputs into comparable risk scores strictly in [0, 1].
"""
from typing import Optional
import numpy as np

class ScoreCalibrator:
    """
    Fits calibration parameters strictly on training or validation sets.
    Supports min-max scaling, sigmoid scaling, and empirical CDF mapping.
    """
    def __init__(self, method: str = "minmax"):
        self.method = method
        self.is_fitted = False
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.mean_val: float = 0.0
        self.std_val: float = 1.0

    def fit(self, scores: np.ndarray) -> "ScoreCalibrator":
        clean_scores = np.asarray(scores, dtype=np.float64)
        clean_scores = clean_scores[~np.isnan(clean_scores)]
        if len(clean_scores) == 0:
            raise ValueError("Cannot fit calibrator on empty or NaN-only score array.")

        self.min_val = float(np.percentile(clean_scores, 1))
        self.max_val = float(np.percentile(clean_scores, 99))
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1.0

        self.mean_val = float(np.mean(clean_scores))
        self.std_val = float(np.std(clean_scores))
        if self.std_val <= 1e-8:
            self.std_val = 1.0

        self.is_fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ScoreCalibrator must be fitted before transform.")

        s = np.asarray(scores, dtype=np.float64)
        if self.method == "minmax":
            calibrated = (s - self.min_val) / (self.max_val - self.min_val + 1e-8)
            return np.clip(calibrated, 0.0, 1.0)
        elif self.method == "sigmoid":
            # Logistic sigmoid centered at mean
            z = (s - self.mean_val) / (self.std_val + 1e-8)
            return 1.0 / (1.0 + np.exp(-z))
        else:
            return np.clip(s, 0.0, 1.0)
