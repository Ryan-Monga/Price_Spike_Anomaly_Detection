"""
Market Data Acquisition Module.
Retrieves historical market data via yfinance with local disk caching and offline dataset support.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from config import RAW_DATA_DIR, RAW_COLUMNS, DEFAULT_ASSETS, SUPPORTED_PERIODS

logger = logging.getLogger(__name__)

class DataFetchError(Exception):
    """Custom exception for market data retrieval failures."""
    pass

class MarketDataFetcher:
    """
    Fetches, validates, and caches historical market data.
    Provides offline demo fallback datasets if network is unavailable.
    """
    def __init__(self, cache_dir: Path = RAW_DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str, period: str) -> Path:
        sanitized_ticker = ticker.replace("^", "").replace("/", "_").replace(":", "_").upper()
        return self.cache_dir / f"{sanitized_ticker}_{period}.csv"

    def fetch_data(
        self,
        ticker: str,
        period: str = "1y",
        force_refresh: bool = False,
        use_offline_fallback: bool = True,
        offline_mode: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetches historical daily market data for a given ticker and period.
        
        Returns:
            Tuple of (DataFrame with standard OHLCV columns, Metadata dictionary)
        """
        ticker = ticker.strip().upper()
        if not ticker:
            raise DataFetchError("Ticker symbol cannot be empty.")
        
        if period not in SUPPORTED_PERIODS:
            raise DataFetchError(f"Unsupported period: {period}. Supported: {SUPPORTED_PERIODS}")

        cache_path = self._get_cache_path(ticker, period)

        if offline_mode:
            logger.info(f"Offline mode active: generating deterministic dataset for {ticker} ({period})...")
            df = self.generate_offline_dataset(ticker, period)
            return df, {
                "ticker": ticker,
                "period": period,
                "source": "offline_mode",
                "cache_path": None,
                "row_count": len(df),
                "start_date": str(df.index.min().date()),
                "end_date": str(df.index.max().date()),
                "last_timestamp": str(df.index.max()),
                "is_cached": False,
                "is_synthetic": True
            }

        # 1. Check local cache first if not force_refresh
        if not force_refresh and cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if self._validate_dataframe(df):
                    metadata = {
                        "ticker": ticker,
                        "period": period,
                        "source": "local_cache",
                        "cache_path": str(cache_path),
                        "row_count": len(df),
                        "start_date": str(df.index.min().date()),
                        "end_date": str(df.index.max().date()),
                        "last_timestamp": str(df.index.max()),
                        "is_cached": True,
                        "is_synthetic": False
                    }
                    logger.info(f"Loaded cached data for {ticker} ({period}) from {cache_path}")
                    return df, metadata
            except Exception as e:
                logger.warning(f"Failed to read cache at {cache_path}: {e}. Retrying live fetch.")

        # 2. Attempt live download via yfinance
        try:
            import yfinance as yf
            logger.info(f"Downloading live data for {ticker} ({period}) from yfinance...")
            ticker_obj = yf.Ticker(ticker)
            raw_df = ticker_obj.history(period=period, interval="1d", auto_adjust=False)

            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"No market data returned for ticker '{ticker}' and period '{period}'.")

            df = self._clean_and_standardize(raw_df)
            
            # Save to cache
            df.to_csv(cache_path)
            
            metadata = {
                "ticker": ticker,
                "period": period,
                "source": "yahoo_finance",
                "cache_path": str(cache_path),
                "row_count": len(df),
                "start_date": str(df.index.min().date()),
                "end_date": str(df.index.max().date()),
                "last_timestamp": str(df.index.max()),
                "is_cached": False,
                "is_synthetic": False
            }
            return df, metadata

        except Exception as e:
            logger.error(f"Error fetching live data for {ticker}: {e}")
            
            # Check if existing cache exists despite force_refresh
            if cache_path.exists():
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    if self._validate_dataframe(df):
                        logger.info(f"Falling back to existing cached dataset for {ticker}.")
                        return df, {
                            "ticker": ticker,
                            "period": period,
                            "source": "local_cache_fallback",
                            "cache_path": str(cache_path),
                            "row_count": len(df),
                            "start_date": str(df.index.min().date()),
                            "end_date": str(df.index.max().date()),
                            "last_timestamp": str(df.index.max()),
                            "is_cached": True,
                            "is_synthetic": False,
                            "warning": f"Network fetch failed ({str(e)}). Serving cached historical data."
                        }
                except Exception:
                    pass

            if use_offline_fallback:
                logger.info(f"Generating offline simulated benchmark dataset for {ticker}...")
                df = self.generate_offline_dataset(ticker, period)
                return df, {
                    "ticker": ticker,
                    "period": period,
                    "source": "offline_simulated",
                    "cache_path": None,
                    "row_count": len(df),
                    "start_date": str(df.index.min().date()),
                    "end_date": str(df.index.max().date()),
                    "last_timestamp": str(df.index.max()),
                    "is_cached": False,
                    "is_synthetic": True,
                    "warning": f"Network/Provider unavailable ({str(e)}). Serving offline benchmark data."
                }
            
            raise DataFetchError(f"Could not retrieve market data for '{ticker}': {e}")

    def _clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names, removes duplicates, handles timezone and missing values."""
        df = df.copy()
        
        # Standardize index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone offset if present for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Match columns case-insensitively
        rename_map = {}
        for col in df.columns:
            for standard in RAW_COLUMNS:
                if col.lower() == standard.lower():
                    rename_map[col] = standard
                    break
        df = df.rename(columns=rename_map)

        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                raise DataFetchError(f"Required column '{col}' is missing from market data.")

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Ensure numeric types
        for col in RAW_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where Close is missing or non-positive
        df = df.dropna(subset=["Close", "Volume"])
        df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]

        # Forward fill any small gaps in Open, High, Low
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill().bfill()

        return df[RAW_COLUMNS]

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validates that DataFrame satisfies minimal market data requirements."""
        if df is None or len(df) < 10:
            return False
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return False
        return True

    def generate_offline_dataset(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Generates realistic geometric Brownian motion market data with stochastic volume
        for offline testing and demonstrations when network is unavailable.
        """
        period_days_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        num_days = period_days_map.get(period, 365)
        
        # Create business dates
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=int(num_days * 1.45))
        dates = pd.bdate_range(start=start_date, end=end_date)[-num_days:]

        # Deterministic seed based on ticker name for repeatability
        seed = abs(hash(ticker)) % (2**31)
        rng = np.random.RandomState(seed)

        initial_price = 150.0 if "BTC" not in ticker else 45000.0
        mu = 0.0004
        sigma = 0.018 if "BTC" not in ticker else 0.035

        # Geometric Brownian Motion
        daily_returns = rng.normal(mu, sigma, size=len(dates))
        price_series = initial_price * np.exp(np.cumsum(daily_returns))

        open_p = price_series * (1 + rng.normal(0, 0.003, size=len(dates)))
        high_p = np.maximum(open_p, price_series) * (1 + np.abs(rng.normal(0, 0.006, size=len(dates))))
        low_p = np.minimum(open_p, price_series) * (1 - np.abs(rng.normal(0, 0.006, size=len(dates))))
        close_p = price_series
        base_vol = 25000000 if "BTC" not in ticker else 15000
        volume = base_vol * np.exp(rng.normal(0, 0.45, size=len(dates)))

        df = pd.DataFrame(
            {
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Close": close_p,
                "Adj Close": close_p,
                "Volume": volume
            },
            index=dates
        )
        return df
