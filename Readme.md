# ChurnGuard — Telco Customer Churn MLOps System

> **Explainable AI + MLflow + FastAPI + Docker CI/CD**

A production-grade customer churn prediction system built for Telco data.  
Supports individual predictions (web UI or REST API) and bulk batch scoring with automated Excel reports.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LOCAL MLOps PIPELINE  (run_pipeline.py)                         │
│                                                                  │
│  1. Data validation                                              │
│  2. Training (3 models) ──► MLflow tracking (dagshub)            │
│  3. Best model ──► MLflow Model Registry (Production)            │
│  4. Download artifact ──► churn_clf.joblib                       │
│  5. Start FastAPI server                                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  DOCKER IMAGE (for deployment)                                   │
│  app/app.py + utils/ + template/ + static/ + *.joblib            │
│                                                                  │
│  Endpoints:                                                      │
│    GET  /           — Web form (individual)                      │
│    POST /predict    — Form submission → HTML result              │
│    POST /api/predict — JSON body → JSON response                 │
│    GET  /batch      — Batch upload page                          │
│    POST /batch_predict — File upload → Excel download            │
│    GET  /api/health — Health check                               │
│    GET  /docs       — Swagger UI                                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  CI/CD (GitHub Actions)                                          │
│  push to Churn-system ──► lint ──► Docker build ──► push   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dataset Schema

Required columns:

| Column | Type | Example |
|--------|------|---------|
| customerID | str | 7590-VHVEG |
| gender | str | Female |
| SeniorCitizen | int | 0 |
| Partner | str | Yes |
| Dependents | str | No |
| tenure | int | 24 |
| PhoneService | str | Yes |
| MultipleLines | str | No |
| InternetService | str | DSL / Fiber optic / No |
| OnlineSecurity | str | Yes / No / No internet service |
| Contract | str | Month-to-month / One year / Two year |
| PaperlessBilling | str | Yes |
| PaymentMethod | str | Electronic check |
| MonthlyCharges | float | 65.50 |
| TotalCharges | float | 1572.00 |
| Churn | str/int | Yes/No or 1/0 |

---

## Local Quick Start

```bash
git clone https://github.com/pritc2611/Customer-Churn-System
cd Churn-system
pip install -r requirements.txt

python run_pipeline.py

python run_pipeline.py --train-only   # training only
python run_pipeline.py --serve-only   # skip training, start API if model 

# 3. Open
#   Web UI  →  http://dagshub.com/<name>/<repo id>.mlflow
#   API docs → http://localhost:8000/docs
```

---

## API Reference

### Individual prediction (JSON)

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.20
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "churn_probability_pct": 62.3,
    "risk_level": "High Risk ⚠️",
    "segment": "Low Engagement / Higher Risk",
    "kpis": {
      "clv": 358.20,
      "clv_tier": "Low",
      "revenue_at_risk": 223.16,
      "expected_ltv": 71.52,
      "monthly_charges": 29.85,
      "net_retention_value": 11.46
    },
    "shap_factors": [
      "Contract is increasing churn risk (impact: +0.312)",
      "tenure is reducing churn risk (impact: -0.187)"
    ]
  }
}
```

### Batch prediction

```bash
# Via curl
curl -X POST http://localhost:8000/batch_predict \
  -F "file=@customers.csv" \
  --output churn_predictions.xlsx

# Via Python
import requests
with open("customers.csv", "rb") as f:
    r = requests.post("http://localhost:8000/batch_predict", files={"file": f})
with open("output.xlsx", "wb") as out:
    out.write(r.content)
```

---

## Models

|       Model           | Tracked in MLflow | Registered       |
|-----------------------|-------------------|------------------|
| Logistic Regression   | ✅               | if best           |
| Random Forest         | ✅               | if best           |
| **Gradient Boosting** | ✅               | ✅ (usually best) |

Segmentation: **KMeans (k=2)** on `[tenure, MonthlyCharges, TotalCharges, ServiceCount]`

---

## Docker

```bash
# Build (after training — artifacts must be present)
docker build -t churnguard .

# Run
docker run -p 8000:8000 churnguard

# Or pull from Docker Hub (after CI push)
docker pull cppd86/churnguard:latest
docker run -p 8000:8000 cppd86/churnguard:latest
```

---

## CI/CD

Push to `Churn-system` branch triggers:
1. **Build** — Docker image with SHA + `latest` tags
2. **Push** — to Docker Hub

Required secrets: `DOCKER_USERNAME`, `DOCKER_PASSWORD`

---

## Business KPIs

| KPI                 | Formula                              |
|---------------------|--------------------------------------|
| CLV                 | `TotalCharges`                       |
| CLV Tier            | qcut into Low / Medium / High        |
| Revenue at Risk     | `CLV × Churn_Probability`            |
| Expected LTV        | `MonthlyCharges × (1/P - 1)`         |
| Net Retention Value | `Expected_LTV × 0.3 − retention_cost`|

---

## Project Structure

```
Churn-system/
├── app/
|    └── app.py               # FastAPI application
|    └── run_pipeline.py      # ML training pipeline
├── training/
     └── training.py                 # Local orchestrator
├── requirements.txt
├── Dockerfile
├── utils/
│   ├── __init__.py
│   └── utils.py              # All helper functions
├── template/
│   ├── page.html             # Input form
│   ├── prediction.html       # Result page
│   └── batch.html            # Batch upload page
├── static/
│   ├── style.css             # Shared styles
│   └── predstyle.css         # Result page styles
├── data/
│   └── telco_churn.csv       # (not committed)
├── shap_background.csv
└── .github/
    └── workflows/
        └── ci-cd.yml
```
