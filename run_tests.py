"""
Script to execute the automated test suite programmatically.
"""
import sys
import pytest

if __name__ == "__main__":
    print("=" * 60)
    print("Running Automated Test Suite for Price Spike Anomaly Detection")
    print("=" * 60)
    args = ["-v", "tests/"]
    exit_code = pytest.main(args)
    sys.exit(exit_code)
