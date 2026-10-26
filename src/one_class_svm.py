"""
One-Class Support Vector Machine Anomaly Detector Module.
Fits an unsupervised boundary on presumed-normal training observations using scikit-learn.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from config import OneClassSVMConfig, CORE_FEATURE_COLUMNS, MODELS_DIR
from src.preprocessing import FeatureScaler

logger = logging.getLogger(__name__)

class ModelIncompatibilityError(Exception):
    """Raised when input feature schema does not match model requirements."""
    pass

class OneClassSVMDetector:
    """
    One-Class SVM detector for market anomalies.
    
    Trained strictly on historical baseline/normal periods.
    Score convention: Larger scores ALWAYS indicate greater anomalousness.
    """
    def __init__(
        self,
        config: Optional[OneClassSVMConfig] = None,
        feature_cols: Optional[List[str]] = None
    ):
        self.config = config or OneClassSVMConfig()
        self.feature_cols = feature_cols if feature_cols is not None else CORE_FEATURE_COLUMNS
        self.scaler = FeatureScaler(scaler_type="robust", feature_cols=self.feature_cols)
        self.model = OneClassSVM(
            kernel=self.config.kernel,
            nu=self.config.nu,
            gamma=self.config.gamma
        )
        self.is_trained = False
        self.training_metadata: Dict[str, Any] = {}
        # Min/max bounds from training set for score calibration to [0, 1]
        self._score_min = -1.0
        self._score_max = 1.0

    def fit(self, train_df: pd.DataFrame) -> "OneClassSVMDetector":
        """
        Fits scaler and One-Class SVM on training DataFrame.
        """
        self._validate_schema(train_df)
        
        # Fit scaler on training observations only
        X_train_scaled = self.scaler.fit_transform(train_df)
        
        logger.info(f"Training One-Class SVM on {len(train_df)} observations with nu={self.config.nu}")
        self.model.fit(X_train_scaled)
        self.is_trained = True

        # In scikit-learn, decision_function is positive for inliers, negative for outliers.
        # Negate so that higher raw score = more anomalous.
        raw_train_scores = -self.model.decision_function(X_train_scaled)
        self._score_min = float(np.percentile(raw_train_scores, 1))
        self._score_max = float(np.percentile(raw_train_scores, 99))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0

        self.training_metadata = {
            "kernel": self.config.kernel,
            "nu": self.config.nu,
            "gamma": self.config.gamma,
            "feature_columns": self.feature_cols,
            "train_observations": len(train_df),
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "score_min": self._score_min,
            "score_max": self._score_max,
            "contamination_warning": "One-Class SVM assumes majority of training data represents normal conditions. Any existing anomalies in training data may introduce boundary contamination."
        }
        return self

    def predict_scores(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers anomaly scores and binary flags.
        
        Returns:
            calibrated_scores: Float array in [0, 1] where higher = more anomalous.
            anomaly_flags: Boolean array (True = anomaly / out-of-boundary).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() or load() before predict.")

        self._validate_schema(df)
        X_scaled = self.scaler.transform(df)

        # Scikit-learn: decision_function > 0 is inlier, < 0 is outlier.
        # Raw score: -decision_function (higher = more anomalous)
        raw_scores = -self.model.decision_function(X_scaled)
        
        # Binary flag: predict == -1 means outside the learned boundary
        raw_preds = self.model.predict(X_scaled)
        flags = (raw_preds == -1)

        # Calibrate raw score into [0, 1] using min-max clipping
        calibrated = (raw_scores - self._score_min) / (self._score_max - self._score_min + 1e-8)
        calibrated_scores = np.clip(calibrated, 0.0, 1.0).astype(np.float64)

        return calibrated_scores, flags

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments input DataFrame with svm_score and svm_anomaly columns.
        """
        df_out = df.copy()
        scores, flags = self.predict_scores(df_out)
        df_out["svm_score"] = scores
        df_out["svm_anomaly"] = flags
        return df_out

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ModelIncompatibilityError(
                f"Input DataFrame is missing required features for One-Class SVM: {missing}"
            )

    def save(self, filepath: Path) -> Path:
        """Saves model, scaler, and metadata to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "config": self.config,
            "metadata": self.training_metadata,
            "_score_min": self._score_min,
            "_score_max": self._score_max
        }
        joblib.dump(payload, filepath)
        logger.info(f"Saved One-Class SVM model to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "OneClassSVMDetector":
        """Loads model, scaler, and metadata from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        try:
            payload = joblib.load(filepath)
            detector = cls(
                config=payload["config"],
                feature_cols=payload["feature_cols"]
            )
            detector.model = payload["model"]
            detector.scaler = payload["scaler"]
            detector.training_metadata = payload.get("metadata", {})
            detector._score_min = payload.get("_score_min", -1.0)
            detector._score_max = payload.get("_score_max", 1.0)
            detector.is_trained = True
            logger.info(f"Successfully loaded One-Class SVM from {filepath}")
            return detector
        except Exception as e:
            raise RuntimeError(f"Failed to load One-Class SVM model from {filepath}: {e}")
