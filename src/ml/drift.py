import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.core.config import settings

logger = logging.getLogger("VitalCore.Drift")

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between baseline (Expected) 
    and operational logs (Actual) to capture population distribution shift.
    
    PSI Thresholds:
    - PSI < 0.1: Stable / No Shift
    - 0.1 <= PSI < 0.2: Moderate Shift
    - PSI >= 0.2: Significant Shift / Drift Detected
    """
    # Remove nulls
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    try:
        # Use quantiles of the expected array to define bin edges
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        
        # Adjust boundary edges slightly to avoid floating point indexing issues
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        
        # Enforce unique bin boundaries (important for highly concentrated columns)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Calculate counts in each bin
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Normalize to fractions (probabilities) and inject tiny epsilon to avoid log(0) / div by 0
        eps = 1e-4
        expected_probs = expected_counts / len(expected)
        actual_probs = actual_counts / len(actual)
        
        expected_probs = np.where(expected_probs == 0, eps, expected_probs)
        actual_probs = np.where(actual_probs == 0, eps, actual_probs)

        # Recalculate fractions to sum to 1.0 after epsilon additions
        expected_probs /= np.sum(expected_probs)
        actual_probs /= np.sum(actual_probs)

        # Compute Population Stability Index
        psi_value = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
        return float(psi_value)
    except Exception as e:
        logger.warning(f"Error calculating PSI: {e}")
        return 0.0

def detect_dataset_drift(baseline_df: pd.DataFrame, serving_df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs comprehensive statistical data drift audit comparing operational serving logs 
    against historical training baseline features.
    
    Runs dual-evaluation:
    1. Kolmogorov-Smirnov (KS) non-parametric hypothesis tests (triggers if p-value < alpha)
    2. Population Stability Index (PSI) (triggers if PSI >= threshold)
    """
    logger.info("🔬 Commencing statistical telemetry drift analysis...")
    
    drift_detected = False
    drift_results: Dict[str, Any] = {}
    
    features_to_check = [
        "heart_rate", "resting_heart_rate", "hrv", "sleep_duration",
        "sleep_quality", "spo2", "steps", "calories_burned",
        "stress_score", "recovery_score", "bmi", "body_temperature",
        "respiratory_rate", "hydration_score"
    ]
    
    # Enforce minimum serving window size to run reliable statistical tests
    if len(serving_df) < 50:
        logger.info(f"Serving window size ({len(serving_df)}) too small to run valid statistical tests (min 50 logs). Skipping drift check.")
        return False, {"message": "Insufficient logs for drift test."}
        
    for feature in features_to_check:
        if feature in baseline_df.columns and feature in serving_df.columns:
            base_vals = baseline_df[feature].values
            serv_vals = serving_df[feature].values
            
            # 1. Run Kolmogorov-Smirnov 2-sample test
            ks_stat, p_value = ks_2samp(base_vals, serv_vals)
            ks_drift = bool(p_value < settings.DRIFT_ALPHA)
            
            # 2. Run Population Stability Index (PSI) test
            psi_val = calculate_psi(base_vals, serv_vals, num_bins=10)
            psi_drift = bool(psi_val >= settings.DRIFT_PSI_THRESHOLD)
            
            # Feature drifted if either condition triggers
            feature_drifted = ks_drift or psi_drift
            
            drift_results[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "ks_drift_detected": ks_drift,
                "psi_value": float(psi_val),
                "psi_drift_detected": psi_drift,
                "drift_detected": feature_drifted
            }
            
            if feature_drifted:
                drift_detected = True
                logger.warning(
                    f"🚨 Drift detected in '{feature}'! "
                    f"KS p-value: {p_value:.6f} ({'DRIFT' if ks_drift else 'STABLE'}), "
                    f"PSI: {psi_val:.4f} ({'DRIFT' if psi_drift else 'STABLE'})"
                )
            else:
                logger.info(f"✅ Feature '{feature}' profile stable. KS p-val: {p_value:.6f}, PSI: {psi_val:.4f}")
                
    return drift_detected, drift_results
