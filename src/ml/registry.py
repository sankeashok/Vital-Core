import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

from src.core.config import settings
from src.core.feature_store import get_feature_store_instance
from src.ml.drift import detect_dataset_drift
from src.ml.train import VitalCoreTrainer

logger = logging.getLogger("VitalCore.Registry")

def run_promotion_pipeline(force: bool = False, simulated_drift: bool = False) -> Dict[str, Any]:
    """
    Main MLOps continuous retraining execution orchestrator:
    1. Ingests raw telemetry logs and training baseline profiles.
    2. Runs statistical Kolmogorov-Smirnov (KS) and Population Stability Index (PSI) drift checks.
    3. Triggers hyperparameter-optimized model retraining if drift is detected or force is True.
    4. Evaluates Candidate (Challenger) validation R² against Active (Champion) model metrics.
    5. Promotes the Challenger via zero-downtime hot-reload updates if performance margins succeed.
    """
    start_time = time.time()
    
    fs = get_feature_store_instance()
    
    # Step 1: Load data from Feature Store
    baseline_df = fs.get_historical_features()
    serving_df = fs.get_serving_logs(limit=1000)
    
    if baseline_df.empty:
        logger.warning("⚠️ Baseline training features missing from offline store. Running data seed...")
        fs.seed_initial_data(3000)
        baseline_df = fs.get_historical_features()

    # Step 2: Detect distribution drift
    drift_detected = False
    drift_details = {}
    
    if not serving_df.empty and len(serving_df) >= 50:
        # Simulate drift by shifting telemetry variables for demonstration purposes
        if simulated_drift:
            logger.warning("🧪 Simulated Drift active! Shifting incoming serving telemetry distributions...")
            serving_df["heart_rate"] += 12.0
            serving_df["hrv"] = np.clip(serving_df["hrv"] - 15.0, 5, 250)
            serving_df["stress_score"] += 20.0
            
        drift_detected, drift_details = detect_dataset_drift(baseline_df, serving_df)
    else:
        logger.info("ℹ️ Serving logs count is below statistical validation size (50 rows). Skipping drift audit.")

    # Step 3: Trigger check
    if not drift_detected and not force:
        logger.info("😴 Statistical data distributions are stable. Retraining is not required.")
        return {
            "status": "skipped",
            "message": "Data distributions are stable. Continuous training skipped.",
            "drift_detected": False,
            "promoted": False,
            "metrics": {}
        }

    logger.info("⚡ Statistical data drift or force trigger detected! Initiating retraining loop...")

    # Combine historical baseline with recent serving telemetry logs to form retrain set
    if not serving_df.empty:
        # Formulate synthetic ground truth target risk score for logs using our clinical logic
        np.random.seed(42)
        y_serving = (
            0.2 * (serving_df["heart_rate"] - 60)
            - 0.3 * (serving_df["hrv"] - 45)
            - 0.2 * (serving_df["sleep_quality"] - 70)
            - 0.5 * (serving_df["sleep_duration"] - 7)
            + 0.3 * serving_df["stress_score"]
            - 0.2 * serving_df["recovery_score"]
            + 0.4 * (serving_df["bmi"] - 22)**2
            + 2.0 * (37.0 - serving_df["body_temperature"])
            + 1.5 * (serving_df["respiratory_rate"] - 12)
            - 0.1 * serving_df["hydration_score"]
        )
        y_serving = np.clip(y_serving + 30.0 + np.random.normal(0, 4, len(serving_df)), 0.0, 100.0)
        serving_df["health_risk_score"] = y_serving
        
        # Merge dataframes
        retrain_df = pd.concat([baseline_df, serving_df], ignore_index=True)
    else:
        retrain_df = baseline_df.copy()

    # Step 4: Run Optuna Training
    trainer = VitalCoreTrainer(n_trials=10) # 10 trials for quick local execution
    try:
        challenger_model, best_params, challenger_metrics = trainer.run_optimization_and_training(retrain_df)
    except Exception as e:
        logger.error(f"❌ Retraining run crashed: {e}")
        return {
            "status": "failed",
            "message": f"Training script failure: {str(e)}",
            "drift_detected": drift_detected,
            "promoted": False,
            "metrics": {}
        }

    # Step 5: Champion vs Challenger Gate
    prod_model_path = os.path.join(settings.MODEL_DIR, "vital_core_production_model.pkl")
    active_manifest_path = settings.MODEL_MANIFEST_PATH
    
    champion_r2 = -float("inf")
    champion_version = "None"
    
    # Parse existing active model details
    if os.path.exists(prod_model_path):
        try:
            active_pkg = joblib.load(prod_model_path)
            champion_r2 = active_pkg.get("metrics", {}).get("r2", -1.0)
            champion_version = active_pkg.get("model_version", "placeholder_v1.0")
            logger.info(f"🏆 Current production Champion model: {champion_version} (R² = {champion_r2:.4f})")
        except Exception as err:
            logger.warning(f"Failed to parse active production Champion metrics: {err}")

    logger.info(f"⚔️ Evaluating Challenger validation score (R² = {challenger_metrics['r2']:.4f}) against Champion...")

    # Enforce promotion rules: Challenger must beat Champion by a statistical delta
    promotion_margin = 0.002
    should_promote = (challenger_metrics["r2"] >= champion_r2 + promotion_margin) or champion_version == "None" or champion_r2 == -1.0
    
    if should_promote:
        logger.info("🚀 Challenger matches or exceeds Champion criteria! Promoting to active Production stage...")
        
        new_version_str = f"retrained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        model_package = {
            "model": challenger_model,
            "model_version": new_version_str,
            "model_type": "random_forest_retrained",
            "metrics": challenger_metrics,
            "best_params": best_params,
            "promoted_at": datetime.utcnow().isoformat()
        }
        
        # Save Challenger pickle locally
        joblib.dump(model_package, prod_model_path)
        
        # Write active manifest details for zero-downtime hot-reloads
        manifest_data = {
            "model_version": new_version_str,
            "model_path": prod_model_path,
            "promoted_at": datetime.utcnow().isoformat(),
            "metrics": challenger_metrics
        }
        with open(active_manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)
            
        logger.info(f"💾 Saved new production model to {prod_model_path} and updated manifest.")
        return {
            "status": "retrained",
            "message": f"New Challenger model ({new_version_str}) promoted. Champion R² ({champion_r2:.4f}) -> Challenger R² ({challenger_metrics['r2']:.4f})",
            "drift_detected": drift_detected,
            "promoted": True,
            "metrics": challenger_metrics
        }
    else:
        logger.info("❌ Challenger model failed to exceed Champion criteria. Rejecting promotion.")
        return {
            "status": "skipped",
            "message": f"Challenger rejected. Challenger R² ({challenger_metrics['r2']:.4f}) is inferior to active Champion ({champion_r2:.4f}).",
            "drift_detected": drift_detected,
            "promoted": False,
            "metrics": challenger_metrics
        }
