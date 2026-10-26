"""
Tests for Synthetic Anomaly Injection Module.
Verifies all 6 controlled anomaly scenarios, preservation of original data,
and ground-truth label generation.
"""
import pytest
import numpy as np
import pandas as pd

from src.anomaly_injection import SyntheticAnomalyInjector

def make_clean_dataset(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(123)
    close = 120.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, size=n)))
    volume = 2000000 * np.exp(np.random.normal(0, 0.25, size=n))
    return pd.DataFrame({
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": volume
    }, index=dates)

def test_all_scenarios_injection():
    injector = SyntheticAnomalyInjector(seed=42)
    clean_df = make_clean_dataset(80)

    for scenario in injector.SCENARIOS:
        orig_copy = clean_df.copy()
        mod_df, gt, record = injector.inject_scenario(
            df=clean_df,
            scenario=scenario,
            duration=3,
            strength=1.2
        )

        # Original input DataFrame must remain UNTOUCHED
        pd.testing.assert_frame_equal(clean_df, orig_copy)

        assert len(mod_df) == len(clean_df)
        assert len(gt) == len(clean_df)
        assert record.scenario == scenario

        # For the first 5 scenarios, ground truth should have positive labels
        if scenario != "high_volatility_normal":
            assert gt.sum() > 0
            assert record.ground_truth_label == 1
        else:
            # Benign regime should have ground truth = 0
            assert gt.sum() == 0
            assert record.ground_truth_label == 0

def test_pump_and_dump_reversal_structure():
    injector = SyntheticAnomalyInjector(seed=99)
    df = make_clean_dataset(80)
    mod_df, gt, record = injector.inject_scenario(
        df=df,
        scenario="pump_and_dump_reversal",
        start_idx=40,
        duration=4,
        strength=1.5
    )
    assert gt.sum() == 4
    # Close price on final dump day must fall back down
    assert record.modified_close[-1] < record.modified_close[-2]
