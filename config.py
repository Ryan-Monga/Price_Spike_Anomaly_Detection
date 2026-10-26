"""
Configuration settings for Price Spike Anomaly Detection for Pump-and-Dump Schemes.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
DB_PATH = BASE_DIR / "data" / "surveillance.db"

# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Assets & Time horizons
DEFAULT_ASSETS: List[str] = ["AAPL", "TSLA", "MSFT", "BTC-USD", "ETH-USD"]
SUPPORTED_PERIODS: List[str] = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# Feature engineering columns
RAW_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

CORE_FEATURE_COLUMNS: List[str] = [
    "log_return",
    "simple_return",
    "volume_change",
    "volatility_30",
    "rolling_mean_30",
    "rolling_std_30",
    "return_zscore",
    "relative_volume",
    "abs_return",
    "hl_range",
]

# Chronological split defaults
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# Default Rolling Window
DEFAULT_ROLLING_WINDOW: int = 30

@dataclass
class BaselineConfig:
    zscore_threshold: float = 2.5
    relative_volume_threshold: float = 2.0
    abs_return_threshold: float = 0.05
    rolling_window: int = DEFAULT_ROLLING_WINDOW

@dataclass
class OneClassSVMConfig:
    kernel: str = "rbf"
    nu: float = 0.05
    gamma: str = "scale"
    random_state: int = 42

@dataclass
class LSTMAutoencoderConfig:
    sequence_length: int = 30
    hidden_dim: int = 32
    latent_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 25
    seed: int = 42

@dataclass
class HybridDetectorConfig:
    w_stat: float = 0.333
    w_svm: float = 0.333
    w_lstm: float = 0.334
    threshold: float = 0.60  # Calibrated anomaly threshold

@dataclass
class GANConfig:
    latent_dim: int = 16
    hidden_dim: int = 32
    sequence_length: int = 30
    num_features: int = len(CORE_FEATURE_COLUMNS)
    learning_rate: float = 0.0002
    epochs: int = 30
    batch_size: int = 16
    seed: int = 42
