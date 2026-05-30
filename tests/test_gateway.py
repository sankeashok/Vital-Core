import pytest
from fastapi.testclient import TestClient

from src.main import app

@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture that boots within lifespan block to load model params."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Verify health status reports successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "Vital-Core"
    assert "model_status" in data

def test_predict_risk_endpoint(client):
    """Verify risk predictions respond with complete schemas and valid risk bounds."""
    payload = {
        "user_id": "test-user-uuid-123",
        "heart_rate": 75.0,
        "resting_heart_rate": 62.0,
        "hrv": 48.0,
        "sleep_duration": 7.5,
        "sleep_quality": 82.0,
        "spo2": 97.5,
        "steps": 7200,
        "calories_burned": 2100.0,
        "stress_score": 35.0,
        "recovery_score": 80.0,
        "bmi": 22.8,
        "body_temperature": 36.6,
        "respiratory_rate": 15.0,
        "hydration_score": 85.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_risk_score" in data
    assert data["risk_category"] in ["Low", "Medium", "High"]
    assert data["confidence_score"] > 0.0
    assert "model_version" in data
    assert data["processing_time_ms"] >= 0.0

def test_predict_risk_input_bounds(client):
    """Verify that Pydantic bounds checking rejects out-of-range sensor values."""
    payload = {
        "user_id": "test-user-uuid-123",
        "heart_rate": 450.0,  # Physically impossible HR - should be rejected (max 220)
        "resting_heart_rate": 62.0,
        "hrv": 48.0,
        "sleep_duration": 7.5,
        "sleep_quality": 82.0,
        "spo2": 97.5,
        "steps": 7200,
        "calories_burned": 2100.0,
        "stress_score": 35.0,
        "recovery_score": 80.0,
        "bmi": 22.8,
        "body_temperature": 36.6,
        "respiratory_rate": 15.0,
        "hydration_score": 85.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic ValidationError

def test_clinical_copilot_endpoint(client):
    """Verify that Clinical Copilot responds successfully with insights."""
    payload = {
        "user_id": "test-user-uuid-123",
        "telemetry_summary": {
            "heart_rate": 85.0,
            "hrv": 22.0,
            "stress_score": 75.0,
            "spo2": 94.0
        },
        "risk_category": "High",
        "predicted_risk_score": 78.5
    }
    
    response = client.post("/copilot", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["anonymized_user_id"] == "test-user-uuid-123"
    assert "clinical_insight" in data
    assert "source_model" in data
    assert len(data["clinical_insight"]) > 50
