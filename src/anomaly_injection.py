"""
Synthetic Anomaly Injection Module.
Injects controlled market manipulation scenarios into financial time series
while preserving untouched originals and generating ground-truth labels for evaluation.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.features import compute_market_features

@dataclass
class AnomalyEventRecord:
    scenario: str
    start_timestamp: str
    end_timestamp: str
    duration_days: int
    strength: float
    ground_truth_label: int  # 1 for anomaly, 0 for normal benchmark
    original_close: List[float]
    modified_close: List[float]
    original_volume: List[float]
    modified_volume: List[float]
    description: str

class SyntheticAnomalyInjector:
    """
    Simulates 6 controlled financial anomaly scenarios:
    1. Sudden upward price shock with volume surge
    2. Sudden downward price shock with volume surge
    3. Multi-day pump and dump with sharp reversal
    4. Heavy volume churn with muted price movement
    5. Extreme return shock without volume confirmation
    6. High-volatility benign regime (for false-positive testing)
    """
    SCENARIOS = [
        "sudden_upward_shock",
        "sudden_downward_shock",
        "pump_and_dump_reversal",
        "volume_churn",
        "extreme_return_no_volume",
        "high_volatility_normal"
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def inject_scenario(
        self,
        df: pd.DataFrame,
        scenario: str,
        start_idx: Optional[int] = None,
        duration: int = 3,
        strength: float = 1.0
    ) -> Tuple[pd.DataFrame, pd.Series, AnomalyEventRecord]:
        """
        Injects a single scenario into a copy of df and updates engineered features.
        
        Args:
            df: Clean market DataFrame with Open, High, Low, Close, Volume.
            scenario: Name of scenario from SCENARIOS.
            start_idx: Index position to inject. If None, chosen randomly in safe range.
            duration: Number of affected trading days.
            strength: Multiplier scaling the severity (1.0 = standard realistic anomaly).
            
        Returns:
            df_modified: DataFrame with updated OHLCV and re-engineered causal features.
            ground_truth: Binary series (1 = injected anomaly window, 0 = normal).
            record: AnomalyEventRecord with audit details.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {self.SCENARIOS}")
        
        n = len(df)
        if n < duration + 2:
            raise ValueError(f"DataFrame too short ({n}) for injection with duration {duration}.")

        min_start = max(0, min(10, n // 5))
        max_start = max(min_start, n - duration - 1)

        if start_idx is None:
            start_pos = self.rng.randint(min_start, max_start + 1) if max_start > min_start else min_start
        else:
            start_pos = max(0, min(start_idx, max_start))

        end_pos = start_pos + duration
        target_dates = df.index[start_pos:end_pos]

        df_mod = df.copy()
        ground_truth = pd.Series(0, index=df.index, dtype=int)

        orig_close = df.loc[target_dates, "Close"].tolist()
        orig_vol = df.loc[target_dates, "Volume"].tolist()

        is_true_anomaly = 1

        if scenario == "sudden_upward_shock":
            # Rapid price spike + large volume surge
            # e.g. +12% to +25% return, 3.5x to 6x volume
            price_mult = 1.0 + (0.15 * strength)
            vol_mult = 1.0 + (3.5 * strength)
            
            df_mod.loc[target_dates[0], "Close"] *= price_mult
            df_mod.loc[target_dates[0], "High"] = max(df_mod.loc[target_dates[0], "High"], df_mod.loc[target_dates[0], "Close"]) * 1.02
            df_mod.loc[target_dates[0], "Volume"] *= vol_mult
            
            ground_truth.loc[target_dates[0]] = 1
            desc = f"Single-day upward price spike (+{((price_mult - 1) * 100):.1f}%) with {vol_mult:.1f}x volume surge."

        elif scenario == "sudden_downward_shock":
            # Rapid selloff / crash + large panic volume surge
            price_mult = 1.0 - min(0.35, 0.14 * strength)
            vol_mult = 1.0 + (4.0 * strength)

            df_mod.loc[target_dates[0], "Close"] *= price_mult
            df_mod.loc[target_dates[0], "Low"] = min(df_mod.loc[target_dates[0], "Low"], df_mod.loc[target_dates[0], "Close"]) * 0.98
            df_mod.loc[target_dates[0], "Volume"] *= vol_mult

            ground_truth.loc[target_dates[0]] = 1
            desc = f"Single-day downward flash dump (-{((1 - price_mult) * 100):.1f}%) with {vol_mult:.1f}x volume spike."

        elif scenario == "pump_and_dump_reversal":
            # Multi-day artificial ramp followed by precipitous cliff reversal
            pump_days = max(2, duration - 1)
            base_p = df_mod.loc[target_dates[0], "Close"]
            
            # Pump phase: cumulative climb with accelerating volume
            for d in range(pump_days):
                day_target = target_dates[d]
                cum_pct = (0.08 * (d + 1) * strength)
                df_mod.loc[day_target, "Close"] = base_p * (1.0 + cum_pct)
                df_mod.loc[day_target, "High"] = df_mod.loc[day_target, "Close"] * 1.03
                df_mod.loc[day_target, "Volume"] *= (2.0 + d * 1.5 * strength)
                ground_truth.loc[day_target] = 1

            # Dump phase on final day: sharp gap down back to or below initial base
            dump_day = target_dates[-1]
            df_mod.loc[dump_day, "Close"] = base_p * (0.95 - 0.05 * strength)
            df_mod.loc[dump_day, "Low"] = df_mod.loc[dump_day, "Close"] * 0.96
            df_mod.loc[dump_day, "Volume"] *= (4.5 * strength)
            ground_truth.loc[dump_day] = 1
            desc = f"Multi-day pump phase (+{(0.08 * pump_days * strength * 100):.1f}%) followed by sharp dump reversal below baseline."

        elif scenario == "volume_churn":
            # Heavy wash-trading / abnormal volume churn without proportional price movement
            vol_mult = 1.0 + (4.5 * strength)
            for dt in target_dates:
                df_mod.loc[dt, "Volume"] *= vol_mult
                # Minimal price change
                df_mod.loc[dt, "Close"] *= (1.0 + self.rng.uniform(-0.005, 0.005))
                ground_truth.loc[dt] = 1
            desc = f"Abnormal volume surge ({vol_mult:.1f}x) across {duration} days with stagnant price change (<0.5%)."

        elif scenario == "extreme_return_no_volume":
            # Large price movement with thin/normal volume (e.g. illiquid gap, bad tick)
            price_mult = 1.0 + (0.16 * strength)
            df_mod.loc[target_dates[0], "Close"] *= price_mult
            df_mod.loc[target_dates[0], "High"] = max(df_mod.loc[target_dates[0], "High"], df_mod.loc[target_dates[0], "Close"])
            # Volume remains baseline or slightly below
            ground_truth.loc[target_dates[0]] = 1
            desc = f"Severe unconfirmed price gap (+{((price_mult - 1) * 100):.1f}%) without corresponding volume surge."

        elif scenario == "high_volatility_normal":
            # Benign high-volatility regime. Ground-truth label is ZERO (0)!
            # Used specifically to test false positive resistance.
            is_true_anomaly = 0
            for dt in target_dates:
                jitter = self.rng.normal(0, 0.025 * strength)
                df_mod.loc[dt, "Close"] *= (1.0 + jitter)
                df_mod.loc[dt, "High"] = max(df_mod.loc[dt, "High"], df_mod.loc[dt, "Close"]) * 1.01
                df_mod.loc[dt, "Low"] = min(df_mod.loc[dt, "Low"], df_mod.loc[dt, "Close"]) * 0.99
                df_mod.loc[dt, "Volume"] *= self.rng.uniform(1.2, 1.8)
                # ground_truth remains 0
            desc = "Controlled benign high-volatility regime to evaluate detector false-positive rate."

        mod_close = df_mod.loc[target_dates, "Close"].tolist()
        mod_vol = df_mod.loc[target_dates, "Volume"].tolist()

        # Re-compute causal technical features on the modified time series
        df_mod = compute_market_features(df_mod)

        record = AnomalyEventRecord(
            scenario=scenario,
            start_timestamp=str(target_dates[0].date()),
            end_timestamp=str(target_dates[-1].date()),
            duration_days=duration,
            strength=strength,
            ground_truth_label=is_true_anomaly,
            original_close=orig_close,
            modified_close=mod_close,
            original_volume=orig_vol,
            modified_volume=mod_vol,
            description=desc
        )

        return df_mod, ground_truth, record
