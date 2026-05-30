import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Enterprise settings management using Pydantic Settings.
    Ensures safe casting of environment variables, default management,
    and type safety across production and local execution stages.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Configuration
    APP_NAME: str = "Vital-Core"
    APP_ENV: str = Field(default="development", env="APP_ENV")
    HOST: str = "0.0.0.0"
    PORT: int = 7860

    # Directory Paths
    BASE_DIR: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    DATA_DIR: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data")))
    MODEL_DIR: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))

    # Database Configuration
    DATABASE_PATH: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/feature_store.db")))

    # MLflow Registry Configuration
    MLFLOW_TRACKING_URI: str = Field(default_factory=lambda: f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mlflow.db'))}")
    MLFLOW_EXPERIMENT_NAME: str = "vital-core-wearable-analytics"

    # Clinical Copilot & Hugging Face Serverless Configuration
    HF_TOKEN: Optional[str] = Field(default=None, env="HF_TOKEN")
    HF_MODEL_ID: str = Field(default="Qwen/Qwen2.5-Coder-32B-Instruct", env="HF_MODEL_ID")

    # Production Model Manifest Path
    MODEL_MANIFEST_PATH: str = Field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/active_model_manifest.json")))

    # Ingestion & Retraining Configurations
    DRIFT_ALPHA: float = 0.05
    DRIFT_PSI_THRESHOLD: float = 0.2
    GOLD_VALIDATION_SIZE: float = 0.2

    def setup_directories(self) -> None:
        """Create necessary directories at startup if they do not exist."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "processed"), exist_ok=True)

# Instantiate settings singleton
settings = Settings()
# Execute directory creation on module load
settings.setup_directories()
