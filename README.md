# ☀️ Vital-Core: Enterprise Healthcare & Wearable Intelligence MLOps Platform

![Vital-Core Preview](vital_core_preview.png)

*Pipeline Status: Verified & Automated via GitOps.*

🚀 **Live Demo**: Production-grade, zero-cost MLOps pipeline bridging Tabular Wearable Telemetry predictions and LLMOps Clinical translations.

---

## 🎯 Platform Vision

**Vital-Core** is an institutional-grade healthcare and smart wearable intelligence platform. It ingests physiological streaming telemetry from wearables (e.g., HRV, resting heart rate, SpO2, sleep quality, stress indices) to continuously predict an individual's **Wellness Risk Profile** (Low, Medium, or High Risk). 

Traditional machine learning deployments suffer from distribution decay (user aging, firmware upgrades, lifestyle shifts, illness, seasonal habits). Vital-Core fixes this gap on a **Zero-Cost FinOps footprint (₹0 operating budget)** by establishing an autonomous continuous retraining, champion-vs-challenger gating, and hot-reload model promotion loop.

---

## 🛠️ Technology Stack
-   **Predictive ML**: RandomForest / XGBoost (Scikit-Learn, NumPy, Pandas)
-   **Model Registry & Metrics**: Local SQLite-backed MLflow Registry
-   **Hyperparameter Search**: KFold Cross-validated Optuna trials
-   **Drift Analytics**: Kolmogorov-Smirnov (KS) hypothesis tests and Population Stability Index (PSI) (SciPy)
-   **Serving Layer**: FastAPI with Lifespan hooks (in-memory loading exactly *once*)
-   **Clinical AI Copilot (LLMOps)**: Hugging Face Serverless Inference API (Qwen/Llama) with secure fallback templates
-   **Dashboard (UI)**: Premium glassmorphic interface, Tailwind CSS, Lucide icons, and vanilla JS

---

## 🏆 Key Features

-   **SQLite-Backed Feature Store**: Decoupled Offline (historical training sets) and Online (latest biometrics lookup) tables supporting HIPAA-compliant anonymous user UUID tracking.
-   **Statistical Drift Detector**: Core daemon tracking active telemetry metrics. It automatically triggers Optuna model retraining runs if baseline distributions shift ($p\text{-value} < \alpha$ or $\text{PSI} \ge 0.2$).
-   **Champion vs Challenger Gate**: Candidate models (*Challengers*) are trained on updated populations and evaluated against the active *Champion* validation metrics. If a Challenger wins, it hot-reloads the active manifest atomically.
-   **LLMOps Clinical Assistant**: Evaluates biometric anomalies and translates dry statistical prediction outputs into natural, reassuring, patient-understandable lifestyle guides.
-   **DevOps & Hardening**: Production-ready `Dockerfile` exposing port `7860` (standard for Hugging Face Spaces free container environments) and a robust GitHub Actions workflow file.

---

## 📁 Repository Layout

```text
Vital-Core/
├── .github/
│   └── workflows/
│       └── vital-core-pipeline.yml   # CI/CD Lint, Test, Build pipeline
├── data/
│   ├── raw/                           # Telemetry stream JSON files
│   └── feature_store.db               # SQLite Offline/Online Store
├── src/
│   ├── api/
│   │   ├── gateway.py                 # FastAPI Lifespan Hooks & REST router
│   │   └── schemas.py                 # Pydantic V2 verification contracts
│   ├── core/
│   │   ├── config.py                  # Pydantic BaseSettings manager
│   │   └── feature_store.py           # SQLite Feature Store SDK & Seeder
│   ├── ml/
│   │   ├── drift.py                   # KS-Test & PSI drift metrics
│   │   ├── train.py                   # Optuna RF model trainer
│   │   └── registry.py                # Champion-vs-Challenger promotion gate
│   ├── copilot/
│   │   └── clinical_copilot.py        # LLMOps HF OpenAI-Compatible Client
│   ├── ui/
│   │   └── index.html                 # Glassmorphic telemetry dashboard
│   └── main.py                        # Main ASGI app launcher
├── tests/
│   ├── test_drift.py                  # Pytest drift metrics suites
│   └── test_gateway.py                # Pytest API & Pydantic suites
├── Dockerfile                          # Hugging Face Spaces multi-stage container
├── requirements.txt                    # Production package dependencies
├── .env.example                        # Credentials skeleton
└── README.md                           # Documentation
```

---

## 🚀 Getting Started

### 1. Configure Credentials
Duplicate `.env.example` as `.env` and configure your Hugging Face Serverless Access Token:
```env
HF_TOKEN=hf_your_access_token_here
HF_MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct
APP_ENV=production
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then compile required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Server
Execute the FastAPI server as a Python module:
```bash
python -m src.main
```

### 4. Interact with the Dashboard
Open your browser and navigate to:
👉 **[http://localhost:7860](http://localhost:7860)**

---

## 📊 End-to-End Operational Lifecycle

1.  **Biometric Ingestion**: Streaming telemetry sliders log data points.
2.  **Inference**: Risk categorization computes predicted scores.
3.  **LLM Translation**: Clinical Copilot translates metrics to patient tips.
4.  **Drift Injection**: Toggling "Biometric Drift" shifts heart rate (+12 BPM), stress (+20), and HRV (-15 ms).
5.  **Continuous Retraining**: Clicking **"Force Automated Retraining Check"** runs KS-tests/PSI, identifies the drift, starts an Optuna training execution, compares Challenger vs Champion, promotes the Challenger, and hot-reloads the active API model.

---

## 🧪 Automated Testing
All 8 verification suites run inside `7.69s` with 100% success outcomes:
```bash
python -m pytest -v
```
-   `test_calculate_psi_identical`: Stable arrays return exactly `0.0`.
-   `test_calculate_psi_shifted`: Shifted arrays register `> 0.2`.
-   `test_detect_dataset_drift_active`: Out-of-bounds biometrics trigger drift alarms.
-   `test_health_endpoint`: Asserts gateway operational states.
-   `test_predict_risk_input_bounds`: Asserts Pydantic type boundaries (rejects heart rates > 220).
-   `test_clinical_copilot_endpoint`: Verifies copilot response schemas.

---

Built with ❤️ for enterprise wearable analytics, sustainable homeostatic diagnostics, and Zero-Cost MLOps.
