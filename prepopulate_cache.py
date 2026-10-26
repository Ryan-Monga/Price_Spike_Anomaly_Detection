"""
Pre-population utility to generate realistic offline cached datasets for default assets.
"""
from pathlib import Path
from src.data_fetcher import MarketDataFetcher
from config import RAW_DATA_DIR, DEFAULT_ASSETS, SUPPORTED_PERIODS

def prepopulate_cache():
    fetcher = MarketDataFetcher(cache_dir=RAW_DATA_DIR)
    for ticker in ["AAPL", "TSLA", "BTC-USD"]:
        for period in ["1y", "6mo"]:
            cache_file = fetcher._get_cache_path(ticker, period)
            if not cache_file.exists():
                df = fetcher.generate_offline_dataset(ticker, period)
                df.to_csv(cache_file)
                print(f"Prepopulated cached dataset for {ticker} ({period}) at {cache_file}")

if __name__ == "__main__":
    prepopulate_cache()
