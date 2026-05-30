import pytest
import numpy as np
import pandas as pd
from src.ml.drift import calculate_psi, detect_dataset_drift

def test_calculate_psi_identical():
    """Verify that Population Stability Index (PSI) is zero for identical arrays."""
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=10.0, size=1000)
    actual = expected.copy()
    
    psi_val = calculate_psi(expected, actual, num_bins=10)
    assert psi_val == 0.0

def test_calculate_psi_shifted():
    """Verify that Population Stability Index (PSI) is high for significantly shifted arrays."""
    np.random.seed(42)
    expected = np.random.normal(loc=100.0, scale=10.0, size=1000)
    actual = np.random.normal(loc=115.0, scale=10.0, size=1000) # Shift right
    
    psi_val = calculate_psi(expected, actual, num_bins=10)
    assert psi_val >= 0.2  # Significant shift threshold

def test_detect_dataset_drift_stable():
    """Verify that dataset drift detector registers stable when distributions match."""
    np.random.seed(42)
    features = ["heart_rate", "hrv", "stress_score"]
    
    base_data = {feat: np.random.normal(50.0, 5.0, 100) for feat in features}
    serv_data = {feat: base_data[feat].copy() for feat in features}
    
    baseline_df = pd.DataFrame(base_data)
    serving_df = pd.DataFrame(serv_data)
    
    drift_detected, results = detect_dataset_drift(baseline_df, serving_df)
    
    # Since they are from the same distribution, drift should not trigger
    assert not drift_detected

def test_detect_dataset_drift_active():
    """Verify that dataset drift detector triggers active status under shifted populations."""
    np.random.seed(42)
    features = [
        "heart_rate", "resting_heart_rate", "hrv", "sleep_duration",
        "sleep_quality", "spo2", "steps", "calories_burned",
        "stress_score", "recovery_score", "bmi", "body_temperature",
        "respiratory_rate", "hydration_score"
    ]
    
    # Normal base
    baseline_df = pd.DataFrame({feat: np.random.normal(50.0, 2.0, 100) for feat in features})
    
    # Highly drifted serving set (elevate stress, lower HRV)
    serving_df = pd.DataFrame({feat: np.random.normal(50.0, 2.0, 100) for feat in features})
    serving_df["stress_score"] += 15.0
    serving_df["hrv"] -= 10.0
    
    drift_detected, results = detect_dataset_drift(baseline_df, serving_df)
    
    assert drift_detected
    assert results["stress_score"]["drift_detected"]
    assert results["hrv"]["drift_detected"]
