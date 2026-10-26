"""
Feature Engineering Module for Market Anomaly Detection.
Computes technical and statistical features strictly using past observations to prevent lookahead leakage.
"""
from typing import List, Optional
import numpy as np
import pandas as pd

from config import CORE_FEATURE_COLUMNS, DEFAULT_ROLLING_WINDOW

EPSILON = 1e-8

def compute_market_features(
    df: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_periods: int = 5
) -> pd.DataFrame:
    """
    Computes rolling return, volatility, volume, and range features.
    
    All calculations are strictly causal (backward-looking) using past observations only.
    Handles division by zero, non-positive prices, and infinite values cleanly.
    
    Args:
        df: DataFrame containing Open, High, Low, Close, Volume with DatetimeIndex.
        rolling_window: Lookback window size for rolling statistics (default: 30).
        min_periods: Minimum observations needed for rolling calculation.
        
    Returns:
        DataFrame with original OHLCV plus engineered CORE_FEATURE_COLUMNS.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty or None.")
    
    for col in ["Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for feature calculation.")

    df_out = df.copy()

    # Safe price and volume series
    close = df_out["Close"].astype(float)
    prev_close = close.shift(1)
    
    high = df_out["High"].astype(float) if "High" in df_out.columns else close
    low = df_out["Low"].astype(float) if "Low" in df_out.columns else close
    volume = df_out["Volume"].astype(float)
    prev_volume = volume.shift(1)

    # 1. Simple return: (Close_t - Close_{t-1}) / Close_{t-1}
    # Avoid zero division
    simple_return = (close - prev_close) / np.where(prev_close > 0, prev_close, np.nan)
    simple_return = simple_return.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df_out["simple_return"] = simple_return

    # 2. Log return: log(Close_t / Close_{t-1})
    price_ratio = np.where((prev_close > 0) & (close > 0), close / prev_close, 1.0)
    log_return = np.log(np.maximum(price_ratio, EPSILON))
    df_out["log_return"] = pd.Series(log_return, index=df_out.index).fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 3. Volume percentage change: (Volume_t - Volume_{t-1}) / (Volume_{t-1} + EPSILON)
    vol_denom = np.where(prev_volume > 0, prev_volume, EPSILON)
    volume_change = (volume - prev_volume) / vol_denom
    df_out["volume_change"] = volume_change.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 4. Rolling mean and standard deviation of simple returns
    rolling_ret = simple_return.rolling(window=rolling_window, min_periods=min_periods)
    rolling_mean_30 = rolling_ret.mean().fillna(0.0)
    rolling_std_30 = rolling_ret.std(ddof=1).fillna(0.0)
    
    df_out["rolling_mean_30"] = rolling_mean_30
    df_out["rolling_std_30"] = rolling_std_30

    # 5. Volatility (rolling standard deviation of returns)
    df_out["volatility_30"] = rolling_std_30

    # 6. Return z-score: (simple_return - rolling_mean) / (rolling_std + EPSILON)
    zscore = (simple_return - rolling_mean_30) / (rolling_std_30 + EPSILON)
    df_out["return_zscore"] = zscore.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 7. Relative volume: Volume_t / (rolling_mean_volume + EPSILON)
    rolling_vol = volume.rolling(window=rolling_window, min_periods=min_periods)
    rolling_mean_vol = rolling_vol.mean().fillna(volume)
    relative_volume = volume / np.maximum(rolling_mean_vol, EPSILON)
    df_out["relative_volume"] = relative_volume.fillna(1.0).replace([np.inf, -np.inf], 1.0)

    # 8. Absolute return
    df_out["abs_return"] = np.abs(simple_return)

    # 9. High-Low relative price range: (High - Low) / (Close + EPSILON)
    hl_diff = np.maximum(high - low, 0.0)
    hl_range = hl_diff / np.maximum(close, EPSILON)
    df_out["hl_range"] = hl_range.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Clip extreme numerical artifacts
    df_out["return_zscore"] = df_out["return_zscore"].clip(-15.0, 15.0)
    df_out["volume_change"] = df_out["volume_change"].clip(-10.0, 50.0)
    df_out["relative_volume"] = df_out["relative_volume"].clip(0.0, 50.0)

    return df_out

def get_feature_matrix(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extracts the clean numeric feature matrix in a strictly ordered schema.
    """
    cols = feature_columns if feature_columns is not None else CORE_FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {missing}")
    
    return df[cols].copy()
