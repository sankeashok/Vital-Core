import logging
import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import optuna
import mlflow
import mlflow.sklearn

from src.core.config import settings

logger = logging.getLogger("VitalCore.Train")

# Silence Optuna spam logs in stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)

class VitalCoreTrainer:
    """
    Enterprise automated training pipeline with Optuna hyperparameter optimization
    and KFold cross-validated evaluation. Logs comprehensive artifacts to MLflow.
    """

    def __init__(self, n_trials: int = 15):
        self.n_trials = n_trials
        self.features = [
            "heart_rate", "resting_heart_rate", "hrv", "sleep_duration",
            "sleep_quality", "spo2", "steps", "calories_burned",
            "stress_score", "recovery_score", "bmi", "body_temperature",
            "respiratory_rate", "hydration_score"
        ]
        self.target = "health_risk_score"

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function for tuning RandomForest Regressor."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
            "n_jobs": -1
        }
        
        # 5-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_va)
            rmse = np.sqrt(mean_squared_error(y_va, preds))
            rmse_scores.append(rmse)
            
        return float(np.mean(rmse_scores))

    def run_optimization_and_training(self, df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, Any], Dict[str, float]]:
        """
        Executes KFold cross-validated Optuna trials to select the best hyperparameter vector,
        trains the final model on the complete input dataset, and tracks execution in MLflow.
        """
        if df.empty or len(df) < 100:
            raise ValueError(f"Input dataframe has insufficient records ({len(df)}) for model training.")
            
        X = df[self.features]
        y = df[self.target]
        
        # Split a standard test set for final reporting
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=settings.GOLD_VALIDATION_SIZE, 
            random_state=42
        )
        
        logger.info(f"🤖 Starting Optuna optimization across {self.n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=self.n_trials)
        
        best_params = study.best_params
        best_cv_rmse = study.best_value
        logger.info(f"🎯 Best Hyperparameters Selected: {best_params} | Best CV RMSE: {best_cv_rmse:.4f}")
        
        # Setup MLflow Experiment and Run
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        
        run_name = f"retrain_rf_{int(time.time())}"
        with mlflow.start_run(run_name=run_name) as run:
            logger.info("💾 Logging parameters and metrics to MLflow experiment tracking registry...")
            
            # Log Optuna parameters
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", "random_forest")
            mlflow.log_param("total_samples", len(df))
            
            # Train final model
            final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            final_model.fit(X_train, y_train)
            
            # Evaluate on held-out gold validation set
            preds = final_model.predict(X_val)
            r2 = r2_score(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            
            # Log metrics
            mlflow.log_metric("cv_rmse", best_cv_rmse)
            mlflow.log_metric("val_r2", r2)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_mae", mae)
            
            # Log feature importances
            for feat, importance in zip(self.features, final_model.feature_importances_):
                mlflow.log_metric(f"importance_{feat}", float(importance))
                
            # Log the Model Artifact using MLflow sklearn module
            mlflow.sklearn.log_model(
                sk_model=final_model, 
                artifact_path="model", 
                registered_model_name="vital_core_rf_regressor"
            )
            
            metrics = {
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "cv_rmse": float(best_cv_rmse)
            }
            
            logger.info(f"✅ Training run completed. Validation R²: {r2:.4f} | RMSE: {rmse:.4f}")
            return final_model, best_params, metrics
