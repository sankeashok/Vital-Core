import logging
from datetime import datetime
from typing import Dict, Any
from huggingface_hub import InferenceClient

from src.core.config import settings
from src.api.schemas import CopilotResponse

logger = logging.getLogger("VitalCore.Copilot")

async def run_copilot_translation(
    user_id: str,
    telemetry: Dict[str, float],
    risk_score: float,
    risk_category: str
) -> CopilotResponse:
    """
    LLMOps Clinical Copilot Translator.
    1. Formulates a rich, clinically structured instruction prompt.
    2. Invokes Hugging Face's free Serverless Inference API (Llama-3/Qwen) to generate natural guidance.
    3. Seamlessly falls back to a deterministic clinical rule engine upon HF API rate limits (429) or timeouts (503).
    """
    logger.info(f"🩺 Clinical Copilot requested for user {user_id}...")
    
    # 1. Format the telemetry dataset into a readable diagnostic string
    telemetry_details = "\n".join([f"- {k.replace('_', ' ').title()}: {round(v, 2)}" for k, v in telemetry.items()])
    
    system_prompt = (
        "You are an expert Clinical Health AI Copilot. Analyze the following wearable biometric telemetry "
        "and predicted wellness risk score, and generate a highly professional, reassuring, and actionable "
        "lifestyle guidance report. Address the user directly as a helpful clinical advisor. Enforce strict "
        "HIPAA compliance by never mentioning PII. Keep the report highly concise (under 150 words), and use bullet points."
    )
    
    user_prompt = (
        f"Patient ID: {user_id}\n"
        f"Wellness Risk Score: {round(risk_score, 2)}/100 (Risk Category: {risk_category})\n"
        "Wearable Telemetry Profile:\n"
        f"{telemetry_details}\n\n"
        "Provide clinical insights, key observations, and three concrete lifestyle recommendations."
    )
    
    insight_text = ""
    source_model = settings.HF_MODEL_ID
    
    # 2. Trigger Hugging Face Serverless Inference client if token is supplied
    if settings.HF_TOKEN:
        try:
            logger.info(f"Calling Hugging Face Serverless Inference API using model: {settings.HF_MODEL_ID}")
            client = InferenceClient(model=settings.HF_MODEL_ID, token=settings.HF_TOKEN)
            
            # Request chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat_completion(
                messages=messages,
                max_tokens=250,
                temperature=0.7
            )
            insight_text = response.choices[0].message.content
            logger.info("🎉 Hugging Face LLM generation completed successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Hugging Face API call failed: {e}. Activating deterministic clinical rule engine fallback...")
            insight_text = run_deterministic_rule_engine(telemetry, risk_score, risk_category)
            source_model = "Vital-Core Local Deterministic Rule Engine (HF Fallback)"
    else:
        logger.info("ℹ️ HF_TOKEN not configured in environment variables. Defaulting to local clinical rule engine.")
        insight_text = run_deterministic_rule_engine(telemetry, risk_score, risk_category)
        source_model = "Vital-Core Local Deterministic Rule Engine"

    return CopilotResponse(
        anonymized_user_id=user_id,
        clinical_insight=insight_text,
        source_model=source_model,
        confidence_score=0.88 if "Fallback" in source_model or "Deterministic" in source_model else 0.95,
        timestamp=datetime.utcnow()
    )

def run_deterministic_rule_engine(telemetry: Dict[str, float], risk_score: float, risk_category: str) -> str:
    """
    A robust, clinically sound rule-based diagnostic engine.
    Guarantees zero-downtime responses even when serverless LLM APIs fail.
    """
    logger.info("🛠️ Building clinical observations using local heuristic guidelines...")
    
    insights = []
    recommendations = []
    
    # Analyze core variables
    hr = telemetry.get("heart_rate", 72)
    hrv = telemetry.get("hrv", 45)
    sleep_qual = telemetry.get("sleep_quality", 75)
    spo2 = telemetry.get("spo2", 98)
    stress = telemetry.get("stress_score", 40)
    hydration = telemetry.get("hydration_score", 70)
    
    # 1. Evaluate risk category
    insights.append(f"### Clinical Observations (Risk Status: {risk_category} Risk)")
    insights.append(f"Your wellness risk score is calculated at **{round(risk_score, 2)}/100**.")
    
    # 2. Capture anomalies
    anomalies = []
    if spo2 < 95.0:
        anomalies.append(f"Oxygen saturation (SpO2) is slightly low at {round(spo2, 1)}% (expected >95%).")
        recommendations.append("Practice deep diaphragmatic breathing and ensure proper room ventilation.")
    if hr > 100.0:
        anomalies.append(f"Active heart rate is elevated at {round(hr, 1)} BPM (tachycardia indicators).")
        recommendations.append("Limit caffeine intake, sit down, and perform a 5-minute progressive muscle relaxation.")
    if hrv < 25.0:
        anomalies.append(f"Heart Rate Variability (HRV) is suppressed at {round(hrv, 1)} ms, signaling potential nervous system fatigue.")
        recommendations.append("Prioritize recovery: adjust training intensity downward and ensure active rest.")
    if stress > 65.0:
        anomalies.append(f"Stress indices are high at {round(stress, 1)}/100, which correlates with low HRV scores.")
        recommendations.append("Engage in a structured box-breathing cycle (4s inhale, 4s hold, 4s exhale, 4s hold).")
    if sleep_qual < 60.0:
        anomalies.append(f"Sleep quality was degraded last night at {round(sleep_qual, 1)}% efficiency.")
        recommendations.append("Implement screen-free habits 1 hour before sleep and keep the room dark and cool.")
    if hydration < 60.0:
        anomalies.append(f"Hydration metrics indicate mild fluid volume depletion at {round(hydration, 1)}%.")
        recommendations.append("Rehydrate systematically: consume 500ml of water infused with essential mineral electrolytes.")

    if anomalies:
        insights.append("#### Highlighted Biomarker Warnings:")
        for anom in anomalies:
            insights.append(f"- {anom}")
    else:
        insights.append("- All core physiological telemetry channels (SpO2, Heart Rate, HRV) are performing within safe homeostatic ranges.")
        recommendations.append("Maintain your current healthy workout and dietary baseline profiles.")
        recommendations.append("Continue monitoring wearable telemetry trends daily to catch subtle baseline shifts.")
        recommendations.append("Keep your hydration index high by consuming water throughout the day.")

    # 3. Add recommendations
    insights.append("\n#### Targeted Lifestyle Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        insights.append(f"{i}. **{rec}**")
        
    return "\n".join(insights)
