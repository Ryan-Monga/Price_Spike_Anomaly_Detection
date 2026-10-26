"""
Statistical Baseline Anomaly Detector Module.
Provides interpretable rule-based anomaly detection using rolling z-scores and volume ratios.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config import BaselineConfig

@dataclass
class BaselineAnomalyResult:
    timestamp: pd.Timestamp
    simple_return: float
    volume_change: float
    relative_volume: float
    statistical_score: float
    is_anomaly: bool
    reasons: List[str]

class StatisticalBaselineDetector:
    """
    Interpretable baseline detector combining return z-scores, volume surges,
    and extreme price-volume discrepancies.
    """
    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.z_thresh = self.config.zscore_threshold
        self.rvol_thresh = self.config.relative_volume_threshold
        self.abs_ret_thresh = self.config.abs_return_threshold

    def detect(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Runs statistical anomaly detection across all rows in df_features.
        
        Args:
            df_features: DataFrame containing simple_return, volume_change,
                         return_zscore, relative_volume, and abs_return.
                         
        Returns:
            DataFrame augmented with:
            - stat_score: continuous score in [0, 1] (higher = more anomalous)
            - stat_anomaly: binary boolean flag
            - stat_reasons: human-readable explanation string
        """
        df = df_features.copy()
        
        # Verify columns exist
        for col in ["simple_return", "volume_change", "return_zscore", "relative_volume", "abs_return"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for statistical detection.")

        z = df["return_zscore"].fillna(0.0).values
        abs_z = np.abs(z)
        rvol = df["relative_volume"].fillna(1.0).values
        ret = df["simple_return"].fillna(0.0).values
        abs_ret = df["abs_return"].fillna(0.0).values
        vol_chg = df["volume_change"].fillna(0.0).values

        n = len(df)
        stat_scores = np.zeros(n, dtype=np.float64)
        is_anomaly = np.zeros(n, dtype=bool)
        reasons_list: List[str] = []

        for i in range(n):
            row_reasons = []
            
            # Sub-scores normalized between 0 and 1
            # Z-score contribution: 0 at z=0, 0.5 at threshold, 1.0 at 2x threshold
            z_score_norm = min(1.0, abs_z[i] / (self.z_thresh * 2.0 + 1e-8))
            
            # Relative volume contribution: 0 at 1.0x, 0.5 at threshold, 1.0 at 2.5x threshold
            rvol_norm = 0.0
            if rvol[i] > 1.0:
                rvol_norm = min(1.0, (rvol[i] - 1.0) / (self.rvol_thresh * 2.0 + 1e-8))

            # Compound volume-price interaction score
            interaction_norm = 0.0
            if rvol[i] >= self.rvol_thresh and abs_z[i] >= self.z_thresh:
                interaction_norm = min(1.0, (abs_z[i] / self.z_thresh) * (rvol[i] / self.rvol_thresh) * 0.25)
            elif rvol[i] >= self.rvol_thresh * 1.5 and abs_ret[i] < 0.015:
                # High volume churn with flat price
                interaction_norm = 0.4

            # Continuous composite score in [0.0, 1.0]
            composite_score = float(np.clip(
                0.40 * z_score_norm + 0.35 * rvol_norm + 0.25 * interaction_norm,
                0.0,
                1.0
            ))
            stat_scores[i] = composite_score

            # Rule evaluation for flags & explanations
            # Rule 1: High positive or negative price shock with abnormal volume
            if abs_z[i] >= self.z_thresh and rvol[i] >= self.rvol_thresh:
                direction = "Positive price surge" if ret[i] > 0 else "Severe price drop"
                row_reasons.append(
                    f"{direction} (z-score: {z[i]:+.2f}, return: {ret[i]:+.2%}) accompanied by abnormal volume ({rvol[i]:.2f}x baseline)."
                )

            # Rule 2: Return is extreme relative to recent volatility alone
            elif abs_z[i] >= self.z_thresh * 1.3:
                row_reasons.append(
                    f"Return magnitude (z-score: {z[i]:+.2f}) is extreme relative to recent 30-day volatility."
                )

            # Rule 3: Extreme volume spike without proportional price movement
            elif rvol[i] >= self.rvol_thresh * 1.8 and abs_ret[i] < 0.02:
                row_reasons.append(
                    f"Abnormally high volume ({rvol[i]:.2f}x baseline) with low price movement ({abs_ret[i]:.2%}), suggesting heavy volume churn."
                )

            # Rule 4: Absolute return exceeds absolute threshold with elevated volume
            elif abs_ret[i] >= self.abs_ret_thresh and rvol[i] >= 1.5:
                row_reasons.append(
                    f"Large absolute daily return ({abs_ret[i]:.2%}) with elevated trading activity ({rvol[i]:.2f}x baseline)."
                )

            if len(row_reasons) > 0:
                is_anomaly[i] = True
                reasons_list.append("; ".join(row_reasons))
            else:
                is_anomaly[i] = False
                reasons_list.append("Normal market behaviour within expected statistical volatility and volume bounds.")

        df["stat_score"] = stat_scores
        df["stat_anomaly"] = is_anomaly
        df["stat_reasons"] = reasons_list
        return df

    def get_summary_stats(self, df_detected: pd.DataFrame) -> Dict[str, Any]:
        """Returns high-level summary metrics of detected baseline anomalies."""
        if "stat_anomaly" not in df_detected.columns:
            return {}
        
        total = len(df_detected)
        anomalies = int(df_detected["stat_anomaly"].sum())
        rate = (anomalies / total) * 100.0 if total > 0 else 0.0
        
        return {
            "total_observations": total,
            "anomaly_count": anomalies,
            "anomaly_rate_pct": round(rate, 2),
            "mean_score": round(float(df_detected["stat_score"].mean()), 4),
            "max_score": round(float(df_detected["stat_score"].max()), 4),
        }
