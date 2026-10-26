"""
Storage and Persistence Module.
Manages local SQLite surveillance database and CSV exports for predictions and experiments.
"""
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Any
import pandas as pd

from config import DB_PATH, PROCESSED_DATA_DIR, EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

class SurveillanceStorage:
    """
    Manages SQLite storage for surveillance records, model registry, and experiment runs.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    asset TEXT,
                    scenario TEXT,
                    model_type TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    f4_score REAL,
                    fpr REAL,
                    true_positives INTEGER,
                    false_positives INTEGER,
                    true_negatives INTEGER,
                    false_negatives INTEGER,
                    duration_days INTEGER,
                    strength REAL,
                    notes TEXT
                )
            """)

            # Alert logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_date TEXT,
                    asset TEXT,
                    price REAL,
                    daily_return REAL,
                    relative_volume REAL,
                    hybrid_score REAL,
                    threshold REAL,
                    detection_status TEXT,
                    reasons TEXT
                )
            """)

            # Model registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    asset TEXT,
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    train_start TEXT,
                    train_end TEXT,
                    filepath TEXT,
                    hyperparameters TEXT
                )
            """)
            conn.commit()

    def log_experiment(self, experiment_data: Dict[str, Any]) -> int:
        """Logs an evaluation experiment run into SQLite."""
        query = """
            INSERT INTO experiments (
                asset, scenario, model_type, precision, recall, f1_score, f4_score,
                fpr, true_positives, false_positives, true_negatives, false_negatives,
                duration_days, strength, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            experiment_data.get("asset", "UNKNOWN"),
            experiment_data.get("scenario", "manual"),
            experiment_data.get("model_type", "Hybrid"),
            float(experiment_data.get("precision", 0.0)),
            float(experiment_data.get("recall", 0.0)),
            float(experiment_data.get("f1_score", 0.0)),
            float(experiment_data.get("f4_score", 0.0)),
            float(experiment_data.get("fpr", 0.0)),
            int(experiment_data.get("true_positives", 0)),
            int(experiment_data.get("false_positives", 0)),
            int(experiment_data.get("true_negatives", 0)),
            int(experiment_data.get("false_negatives", 0)),
            int(experiment_data.get("duration_days", 1)),
            float(experiment_data.get("strength", 1.0)),
            experiment_data.get("notes", "")
        )
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_past_experiments(self, limit: int = 50) -> pd.DataFrame:
        """Retrieves history of past experiments."""
        query = "SELECT * FROM experiments ORDER BY id DESC LIMIT ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df

    def log_alerts(self, df_alerts: pd.DataFrame, asset: str) -> int:
        """Logs detected anomaly rows into SQLite."""
        if df_alerts.empty:
            return 0
        records = []
        for idx, row in df_alerts.iterrows():
            records.append((
                str(idx.date()) if hasattr(idx, "date") else str(idx),
                asset,
                float(row.get("Close", 0.0)),
                float(row.get("simple_return", 0.0)),
                float(row.get("relative_volume", 1.0)),
                float(row.get("hybrid_score", 0.0)),
                float(row.get("hybrid_threshold", 0.6)),
                str(row.get("detection_status", "ANOMALY")),
                str(row.get("alert_explanation", ""))
            ))
        query = """
            INSERT INTO alert_logs (
                event_date, asset, price, daily_return, relative_volume,
                hybrid_score, threshold, detection_status, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def export_predictions_csv(self, df: pd.DataFrame, ticker: str) -> Path:
        """Exports detected predictions to a timestamped CSV."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized = ticker.replace("^", "").replace("/", "_").replace(":", "_").upper()
        out_path = PROCESSED_DATA_DIR / f"{sanitized}_predictions_{timestamp_str}.csv"
        df.to_csv(out_path)
        logger.info(f"Exported predictions CSV to {out_path}")
        return out_path

    def export_experiment_csv(self, df_eval: pd.DataFrame, prefix: str = "eval") -> Path:
        """Exports evaluation results to CSV."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EXPERIMENTS_DIR / f"{prefix}_{timestamp_str}.csv"
        df_eval.to_csv(out_path, index=False)
        logger.info(f"Exported evaluation CSV to {out_path}")
        return out_path
