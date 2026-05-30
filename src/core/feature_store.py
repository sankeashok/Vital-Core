import os
import sqlite3
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.core.config import settings

logger = logging.getLogger("VitalCore.FeatureStore")

class LightweightFeatureStore:
    """
    Enterprise custom SQLite-backed Feature Store.
    Provides decoupled Offline storage (historical data/training sets),
    Online storage (fast key-value lookup of user's latest biometrics),
    and Serving Logs (audit trail of real-time incoming payloads for drift checks).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to SQLite."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        """Setup SQLite schemas for offline, online and serving logs if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. Offline Feature Store: Immutable historical datasets for model retraining
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS offline_feature_store (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        heart_rate REAL NOT NULL,
                        resting_heart_rate REAL NOT NULL,
                        hrv REAL NOT NULL,
                        sleep_duration REAL NOT NULL,
                        sleep_quality REAL NOT NULL,
                        spo2 REAL NOT NULL,
                        steps INTEGER NOT NULL,
                        calories_burned REAL NOT NULL,
                        stress_score REAL NOT NULL,
                        recovery_score REAL NOT NULL,
                        bmi REAL NOT NULL,
                        body_temperature REAL NOT NULL,
                        respiratory_rate REAL NOT NULL,
                        hydration_score REAL NOT NULL,
                        health_risk_score REAL NOT NULL
                    );
                """)
                
                # Create index on user_id and timestamp for quick retrieval
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_offline_user ON offline_feature_store (user_id);")
                
                # 2. Online Feature Store: Holds the absolute latest state of each anonymized user for immediate retrieval
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS online_feature_store (
                        user_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        heart_rate REAL NOT NULL,
                        resting_heart_rate REAL NOT NULL,
                        hrv REAL NOT NULL,
                        sleep_duration REAL NOT NULL,
                        sleep_quality REAL NOT NULL,
                        spo2 REAL NOT NULL,
                        steps INTEGER NOT NULL,
                        calories_burned REAL NOT NULL,
                        stress_score REAL NOT NULL,
                        recovery_score REAL NOT NULL,
                        bmi REAL NOT NULL,
                        body_temperature REAL NOT NULL,
                        respiratory_rate REAL NOT NULL,
                        hydration_score REAL NOT NULL
                    );
                """)

                # 3. Serving Logs: Captures all operational inputs received by /predict endpoint to compute drift metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS serving_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        heart_rate REAL NOT NULL,
                        resting_heart_rate REAL NOT NULL,
                        hrv REAL NOT NULL,
                        sleep_duration REAL NOT NULL,
                        sleep_quality REAL NOT NULL,
                        spo2 REAL NOT NULL,
                        steps INTEGER NOT NULL,
                        calories_burned REAL NOT NULL,
                        stress_score REAL NOT NULL,
                        recovery_score REAL NOT NULL,
                        bmi REAL NOT NULL,
                        body_temperature REAL NOT NULL,
                        respiratory_rate REAL NOT NULL,
                        hydration_score REAL NOT NULL
                    );
                """)
                
                conn.commit()
                logger.info("✅ SQLite Database structures initialized successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to initialize Feature Store schema: {e}")
            raise

    def log_serving_features(self, payload: Dict[str, Any]) -> None:
        """
        Record operational request metrics into the serving logs table in real-time.
        Updates the online feature store with the latest values for the specific user.
        """
        import datetime
        timestamp = datetime.datetime.utcnow().isoformat()
        user_id = payload.get("user_id", "ANON_USER")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert into serving logs
                cursor.execute("""
                    INSERT INTO serving_logs (
                        timestamp, user_id, heart_rate, resting_heart_rate, hrv, 
                        sleep_duration, sleep_quality, spo2, steps, calories_burned, 
                        stress_score, recovery_score, bmi, body_temperature, 
                        respiratory_rate, hydration_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    timestamp, user_id, payload["heart_rate"], payload["resting_heart_rate"], payload["hrv"],
                    payload["sleep_duration"], payload["sleep_quality"], payload["spo2"], payload["steps"],
                    payload["calories_burned"], payload["stress_score"], payload["recovery_score"], payload["bmi"],
                    payload["body_temperature"], payload["respiratory_rate"], payload["hydration_score"]
                ))
                
                # Upsert into online feature store
                cursor.execute("""
                    INSERT INTO online_feature_store (
                        user_id, timestamp, heart_rate, resting_heart_rate, hrv, 
                        sleep_duration, sleep_quality, spo2, steps, calories_burned, 
                        stress_score, recovery_score, bmi, body_temperature, 
                        respiratory_rate, hydration_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        timestamp=excluded.timestamp,
                        heart_rate=excluded.heart_rate,
                        resting_heart_rate=excluded.resting_heart_rate,
                        hrv=excluded.hrv,
                        sleep_duration=excluded.sleep_duration,
                        sleep_quality=excluded.sleep_quality,
                        spo2=excluded.spo2,
                        steps=excluded.steps,
                        calories_burned=excluded.calories_burned,
                        stress_score=excluded.stress_score,
                        recovery_score=excluded.recovery_score,
                        bmi=excluded.bmi,
                        body_temperature=excluded.body_temperature,
                        respiratory_rate=excluded.respiratory_rate,
                        hydration_score=excluded.hydration_score;
                """, (
                    user_id, timestamp, payload["heart_rate"], payload["resting_heart_rate"], payload["hrv"],
                    payload["sleep_duration"], payload["sleep_quality"], payload["spo2"], payload["steps"],
                    payload["calories_burned"], payload["stress_score"], payload["recovery_score"], payload["bmi"],
                    payload["body_temperature"], payload["respiratory_rate"], payload["hydration_score"]
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log serving features for user {user_id}: {e}")

    def get_latest_online_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Query the Online Feature Store for the absolute latest biometrics state of a user.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM online_feature_store WHERE user_id = ?;", (user_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to fetch online features for user {user_id}: {e}")
            return None

    def get_serving_logs(self, limit: int = 1000) -> pd.DataFrame:
        """
        Pull recent serving payloads as a Pandas DataFrame to compute drift statistics.
        """
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM serving_logs ORDER BY id DESC LIMIT ?;", 
                    conn, 
                    params=(limit,)
                )
                if not df.empty:
                    # Drop SQLite metadata column before returning to model consumer
                    df = df.drop(columns=["id", "timestamp", "user_id"], errors="ignore")
                return df
        except Exception as e:
            logger.error(f"Failed to fetch serving logs: {e}")
            return pd.DataFrame()

    def get_historical_features(self) -> pd.DataFrame:
        """
        Fetch the complete immutable offline dataset for retraining runs.
        """
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query("SELECT * FROM offline_feature_store ORDER BY id ASC;", conn)
                if not df.empty:
                    df = df.drop(columns=["id", "timestamp", "user_id"], errors="ignore")
                return df
        except Exception as e:
            logger.error(f"Failed to fetch historical features: {e}")
            return pd.DataFrame()

    def save_offline_features(self, df: pd.DataFrame) -> None:
        """
        Bulk insert processed training data frames back into the offline feature store.
        """
        import datetime
        timestamp = datetime.datetime.utcnow().isoformat()
        try:
            # Add metadata columns if they don't exist
            if "timestamp" not in df.columns:
                df["timestamp"] = timestamp
            if "user_id" not in df.columns:
                # Assign static random UUIDs
                df["user_id"] = [f"seeded-user-{i}" for i in range(len(df))]
                
            with self._get_connection() as conn:
                # SQLite pandas integration
                df.to_sql("offline_feature_store", conn, if_exists="append", index=False)
                logger.info(f"💾 Bulk saved {len(df)} features to offline_feature_store.")
        except Exception as e:
            logger.error(f"Failed to bulk write offline features: {e}")

    def seed_initial_data(self, n_samples: int = 3000, drift_multiplier: float = 1.0) -> None:
        """
        Populate the database with a high-fidelity synthetic baseline training set.
        Supports controlled drift injection to simulate wearable signal anomalies.
        """
        try:
            # Check if already seeded
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM offline_feature_store;")
                count = cursor.fetchone()[0]
                if count >= 1000:
                    logger.info("Database is already seeded with sufficient training baseline features.")
                    return

            logger.info(f"🌱 Seeding database with {n_samples} synthetic wearable baseline telemetry records...")
            np.random.seed(1337)
            
            # 1. Generate physiological attributes matching standard clinical distributions
            heart_rate = np.random.normal(loc=72.0 * drift_multiplier, scale=10.0, size=n_samples)
            resting_heart_rate = np.random.normal(loc=60.0 * drift_multiplier, scale=5.0, size=n_samples)
            hrv = np.random.exponential(scale=45.0 / drift_multiplier, size=n_samples) + 10.0
            sleep_duration = np.random.normal(loc=7.2, scale=1.1, size=n_samples)
            sleep_quality = np.random.uniform(low=40.0, high=95.0, size=n_samples)
            spo2 = np.random.uniform(low=95.0, high=100.0, size=n_samples)
            steps = np.random.negative_binomial(n=10, p=10/(10+6500), size=n_samples)
            calories_burned = steps * 0.04 + np.random.normal(loc=1500.0, scale=200.0, size=n_samples)
            stress_score = np.random.uniform(low=10.0, high=85.0 * drift_multiplier, size=n_samples)
            recovery_score = np.clip(100.0 - stress_score + np.random.normal(0, 10, n_samples), 10, 100)
            bmi = np.random.normal(loc=24.5, scale=4.0, size=n_samples)
            body_temperature = np.random.normal(loc=36.6, scale=0.3, size=n_samples)
            respiratory_rate = np.random.normal(loc=16.0, scale=2.0, size=n_samples)
            hydration_score = np.random.uniform(low=50.0, high=100.0, size=n_samples)
            
            # Enforce range bounds
            heart_rate = np.clip(heart_rate, 40, 180)
            resting_heart_rate = np.clip(resting_heart_rate, 40, 100)
            hrv = np.clip(hrv, 5, 200)
            sleep_duration = np.clip(sleep_duration, 2, 14)
            sleep_quality = np.clip(sleep_quality, 10, 100)
            spo2 = np.clip(spo2, 80, 100)
            stress_score = np.clip(stress_score, 0, 100)
            body_temperature = np.clip(body_temperature, 35.0, 42.0)
            respiratory_rate = np.clip(respiratory_rate, 8, 30)
            
            # 2. Formulate complex clinical target (Health Risk Score: 0 to 100)
            # High risk is caused by high HR + low HRV + high stress + low SpO2 + low sleep
            y = (
                0.2 * (heart_rate - 60)
                - 0.3 * (hrv - 45)
                - 0.2 * (sleep_quality - 70)
                - 0.5 * (sleep_duration - 7)
                + 0.3 * stress_score
                - 0.2 * recovery_score
                + 0.4 * (bmi - 22)**2
                + 2.0 * (37.0 - body_temperature)
                + 1.5 * (respiratory_rate - 12)
                - 0.1 * hydration_score
            )
            # Scale y to 0-100 wellness score range
            y = np.clip(y + 30.0, 0.0, 100.0)
            
            df = pd.DataFrame({
                "heart_rate": heart_rate,
                "resting_heart_rate": resting_heart_rate,
                "hrv": hrv,
                "sleep_duration": sleep_duration,
                "sleep_quality": sleep_quality,
                "spo2": spo2,
                "steps": steps.astype(int),
                "calories_burned": calories_burned,
                "stress_score": stress_score,
                "recovery_score": recovery_score,
                "bmi": bmi,
                "body_temperature": body_temperature,
                "respiratory_rate": respiratory_rate,
                "hydration_score": hydration_score,
                "health_risk_score": y
            })
            
            self.save_offline_features(df)
            logger.info("✅ Database seeded successfully.")
        except Exception as e:
            logger.error(f"Failed to seed initial data: {e}")

# Singleton Pattern for DB Manager
_feature_store_instance: Optional[LightweightFeatureStore] = None

def get_feature_store_instance() -> LightweightFeatureStore:
    """Singleton getter for the Feature Store connector."""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = LightweightFeatureStore(settings.DATABASE_PATH)
    return _feature_store_instance
