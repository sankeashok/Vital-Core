
# Vital-Core — Master Prompt for Enterprise MLOps Architecture

## ROLE

Act as a Principal MLOps Architect, Staff Machine Learning Engineer, Healthcare AI Architect, Data Platform Engineer, SRE, DevOps Engineer, Cloud Architect, Product Architect, and Technical Reviewer.

You are designing a production-grade healthcare AI and wearable intelligence platform named:

# Vital-Core

The objective is to build a flagship MLOps portfolio project demonstrating the complete machine learning lifecycle, including automated retraining and continuous learning.

This project should be designed at enterprise standards and serve as evidence of Staff-level MLOps thinking.

---

## PROJECT VISION

Vital-Core is a healthcare and smart wearable intelligence platform.

It ingests physiological and behavioral signals from wearable devices and continuously predicts an individual's health risk profile.

Examples:

- Elevated cardiovascular risk
- Sleep degradation risk
- Stress risk
- Fatigue risk
- Wellness score

The platform must continuously learn from new data and autonomously improve its models.

The project should demonstrate:

- Machine Learning Engineering
- MLOps
- Monitoring
- Model Governance
- Automated Retraining
- Feature Management
- Cloud Deployment
- Observability

---

## BUSINESS PROBLEM

Wearable health data changes continuously.

Examples:

- User aging
- Lifestyle changes
- Seasonal changes
- Exercise habits
- Illness
- Device firmware changes

These changes cause:

- Data Drift
- Concept Drift
- Prediction Drift

Traditional ML deployments become stale.

Vital-Core must detect these changes and continuously adapt.

---

## PRIMARY GOAL

Design an end-to-end autonomous MLOps platform.

The platform should:

1. Train models
2. Deploy models
3. Monitor models
4. Detect drift
5. Trigger retraining
6. Compare challenger models
7. Promote models automatically
8. Roll back if necessary

---

## DATA SOURCES

Simulate wearable data including:

- Heart Rate
- Resting Heart Rate
- HRV
- Sleep Duration
- Sleep Quality
- SpO2
- Steps
- Calories Burned
- Stress Score
- Recovery Score
- BMI
- Temperature
- Respiratory Rate
- Hydration Score

Target:

- Health Risk Score
or
- High/Medium/Low Risk Classification

---

## REQUIRED ARCHITECTURE SECTIONS

### Data Layer
- Raw Zone
- Processed Zone
- Feature Engineering
- Data Validation
- Data Versioning
- Data Quality Monitoring

### Feature Store
Evaluate:
- Feast
- Custom Feature Store

Design:
- Online Store
- Offline Store
- Feature Versioning
- Feature Lineage

### Training Layer
- Automated Training
- Hyperparameter Tuning
- Cross Validation
- MLflow
- DagsHub
- Reproducibility

### Model Registry
- Development
- Staging
- Production
- Archived
- Promotion Rules
- Rollback Strategy

### Serving Layer
- FastAPI
- Docker
- Cloud Run
- Optional Kubernetes
- Blue/Green Deployment
- Canary Deployment

### Monitoring Layer
Monitor:
- Data Drift
- Concept Drift
- Prediction Drift
- Latency
- Errors
- Resource Usage

Tools:
- Prometheus
- Grafana
- MLflow Metrics
- Evidently AI

### Retraining Layer (Most Important)
Design:

- Scheduled Retraining
- Drift-Based Retraining
- Event-Based Retraining
- Performance-Based Retraining

Threshold Examples:

- PSI > 0.2
- KS p-value < 0.05
- Accuracy drop > 10%

Provide:
- Trigger Flow
- Retraining Flow
- Validation Flow
- Promotion Flow
- Rollback Flow

### Champion vs Challenger
Design:

- Evaluation Metrics
- Statistical Significance Testing
- Promotion Rules
- Rollback Rules
- Shadow Deployment
- A/B Testing

### Orchestration
Compare:
- Airflow
- Dagster
- Prefect
- Kubeflow

Recommend the best option.

### Security
- Secrets Management
- PII Handling
- Encryption
- Authentication
- Authorization
- Audit Trails
- HIPAA-style considerations

### CI/CD
Design GitHub Actions workflow:

- Lint
- Unit Tests
- Integration Tests
- Model Validation
- Docker Build
- Security Scan
- Deployment
- Verification
- Rollback

### Cloud Architecture
Provide:
- Local Development
- Free Tier
- Production Architecture

### Failure Scenarios
Analyze:

- Model Drift
- Failed Retraining
- Feature Corruption
- Deployment Failure
- Registry Failure

Provide mitigations.

---

## DELIVERABLES

Provide:

1. End-to-End Architecture Diagram
2. Component Diagram
3. Sequence Diagram
4. Data Flow Diagram
5. Retraining Flow Diagram
6. Champion-Challenger Diagram
7. CI/CD Diagram
8. Monitoring Architecture
9. Folder Structure
10. Technology Stack
11. MVP Roadmap
12. Production Roadmap
13. Resume Bullet Points
14. Interview Questions
15. Architecture Review Questions

Design everything as if Vital-Core is intended to become a real-world healthcare AI platform demonstrating elite MLOps engineering practices.

---

## BONUS

Add a Synthetic Wearable Data Generator capable of injecting controlled drift so the retraining and governance lifecycle can be demonstrated end-to-end.
