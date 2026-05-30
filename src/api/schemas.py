from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class WearableTelemetryInput(BaseModel):
    """
    Pydantic schema representing clinical-grade physiological wearable input.
    Enforces HIPAA anonymous identifiers and includes strict sensor bounds checking.
    """
    user_id: str = Field(..., description="Anonymized unique UUID representing the patient/wearer.")
    heart_rate: float = Field(..., description="Current heart rate in BPM.", ge=30.0, le=220.0)
    resting_heart_rate: float = Field(..., description="Resting heart rate in BPM.", ge=30.0, le=120.0)
    hrv: float = Field(..., description="Heart Rate Variability in ms.", ge=5.0, le=250.0)
    sleep_duration: float = Field(..., description="Total sleep time in hours.", ge=0.0, le=24.0)
    sleep_quality: float = Field(..., description="Sleep quality index score percentage.", ge=0.0, le=100.0)
    spo2: float = Field(..., description="Oxygen saturation level percentage.", ge=50.0, le=100.0)
    steps: int = Field(..., description="Total steps walked in 24 hours.", ge=0, le=100000)
    calories_burned: float = Field(..., description="Calories burned in kcal.", ge=0.0, le=10000.0)
    stress_score: float = Field(..., description="Stress index level metric.", ge=0.0, le=100.0)
    recovery_score: float = Field(..., description="Body recovery index level metric.", ge=0.0, le=100.0)
    bmi: float = Field(..., description="Body Mass Index value.", ge=10.0, le=60.0)
    body_temperature: float = Field(..., description="Body temperature in degrees Celsius.", ge=30.0, le=45.0)
    respiratory_rate: float = Field(..., description="Respiratory rate in breaths per minute.", ge=5.0, le=50.0)
    hydration_score: float = Field(..., description="Hydration index score percentage.", ge=0.0, le=100.0)

class PredictResponse(BaseModel):
    """
    Standard schema for real-time inference prediction output.
    """
    predicted_risk_score: float = Field(..., description="Calculated Wellness Risk Score (0 to 100).")
    risk_category: str = Field(..., description="Classified risk level: Low, Medium, or High.")
    confidence_score: float = Field(..., description="Inference model prediction confidence (0.0 to 1.0).")
    model_version: str = Field(..., description="Active production model version string.")
    processing_time_ms: float = Field(..., description="Processing time inside prediction server in milliseconds.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC Timestamp of inference request completion.")

class CopilotRequest(BaseModel):
    """
    Schema for clinical copilot translation prompt.
    """
    user_id: str = Field(..., description="Anonymized unique UUID representing the patient.")
    telemetry_summary: Dict[str, float] = Field(..., description="The telemetry key-values to analyze.")
    risk_category: str = Field(..., description="Current predicted risk level classification.")
    predicted_risk_score: float = Field(..., description="The predicted wellness risk score.")

class CopilotResponse(BaseModel):
    """
    Interactive clinical advice output schema from LLM Clinical Copilot.
    """
    anonymized_user_id: str = Field(..., description="Anonymized patient unique UUID.")
    clinical_insight: str = Field(..., description="Human-understandable medical recommendations and observations.")
    source_model: str = Field(..., description="The Hugging Face LLM model used or fallback signature.")
    confidence_score: float = Field(..., description="Clinical insight semantic score.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RetrainResponse(BaseModel):
    """
    Automated Retraining and Model Gating Execution outcome schema.
    """
    status: str = Field(..., description="retrained, skipped, or failed.")
    message: str = Field(..., description="Human-readable description of retraining execution.")
    drift_detected: bool = Field(..., description="True if statistical drift (KS/PSI) triggered retraining.")
    promoted: bool = Field(..., description="True if candidate model was promoted (val metrics improved).")
    metrics: Dict[str, float] = Field(..., description="Validation metrics comparing candidate (Challenger) vs active (Champion) models.")
    duration_seconds: float = Field(..., description="Execution pipeline elapsed time in seconds.")
