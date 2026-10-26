"""
Preprocessing and Data Splitting Module.
Implements strict chronological train/val/test splitting, training-only fitted scaling,
and causal sequence window generation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import (
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    CORE_FEATURE_COLUMNS,
)

@dataclass
class SplitMetadata:
    train_dates: Tuple[str, str]
    val_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    train_size: int
    val_size: int
    test_size: int
    total_size: int

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """
    Performs strictly chronological splitting without shuffling.
    
    Args:
        df: Time-sorted DataFrame with DatetimeIndex.
        train_ratio: Fraction for training split (default: 0.70).
        val_ratio: Fraction for validation split (default: 0.15).
        test_ratio: Fraction for test split (default: 0.15).
        
    Returns:
        train_df, val_df, test_df, and SplitMetadata
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot split empty DataFrame.")
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.3f}")
    
    n = len(df)
    if n < 30:
        raise ValueError(f"Insufficient observations ({n}) for train/val/test chronological split.")
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    metadata = SplitMetadata(
        train_dates=(str(train_df.index.min().date()), str(train_df.index.max().date())),
        val_dates=(str(val_df.index.min().date()), str(val_df.index.max().date())),
        test_dates=(str(test_df.index.min().date()), str(test_df.index.max().date())),
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        total_size=n
    )

    return train_df, val_df, test_df, metadata

class FeatureScaler:
    """
    Fits scaler strictly on training observations and transforms subsequent data.
    """
    def __init__(self, scaler_type: str = "robust", feature_cols: Optional[List[str]] = None):
        self.scaler_type = scaler_type
        self.feature_cols = feature_cols if feature_cols is not None else CORE_FEATURE_COLUMNS
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        X = train_df[self.feature_cols].values
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before transform.")
        X = df[self.feature_cols].values
        return self.scaler.transform(X)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

def create_causal_sequences(
    scaled_data: np.ndarray,
    timestamps: pd.DatetimeIndex,
    sequence_length: int = 30
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Generates causal rolling sequences of shape (N - sequence_length + 1, sequence_length, D).
    Each sequence consists strictly of past observations up to its final observation.
    
    The returned target timestamps correspond to the final observation of each sequence.
    
    Args:
        scaled_data: Array of shape (T, num_features).
        timestamps: DatetimeIndex of length T.
        sequence_length: Number of lookback time-steps.
        
    Returns:
        sequences: Array of shape (N, sequence_length, num_features)
        target_timestamps: DatetimeIndex of length N matching the last step of each sequence.
    """
    t_len, num_features = scaled_data.shape
    if t_len < sequence_length:
        raise ValueError(
            f"Data length ({t_len}) is shorter than sequence length ({sequence_length})."
        )
    
    num_sequences = t_len - sequence_length + 1
    sequences = np.empty((num_sequences, sequence_length, num_features), dtype=np.float32)

    for i in range(num_sequences):
        sequences[i] = scaled_data[i : i + sequence_length]

    # Associated timestamp is the final observation in the sequence window
    target_timestamps = timestamps[sequence_length - 1 :]
    return sequences, target_timestamps
