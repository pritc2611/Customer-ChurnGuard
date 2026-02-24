from __future__ import annotations
import os
import io
import shap
import joblib
import pandas as pd
from contextlib import asynccontextmanager
import mlflow
import dagshub
from fastapi import FastAPI, Form, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from util.utils import (
    CustomerInput,
    generate_reasons,
    get_segment,
    get_risk_tier,
    get_risk_color,
    calculate_individual_kpis,
    read_uploaded_file,
    run_batch_prediction,
    df_to_excel_bytes,
    load_html_template,
    ALL_FEATURES,
    build_service_count,
)


# ─────────────────────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
model           = None
transformer     = None
shap_explainer  = None
seg_bundle      = None   # {"scaler": ..., "kmeans": ...}
full_pipeline   = None
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
SHAPE_DIR = BASE_DIR / "shape-background"
os.environ["MLFLOW_TRACKING_USERNAME"] = "pritc2611"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d69891a2caee7f83dd7ff7cea972fc432996f030"
REGISTERED_MODEL = "TelcoChurnModel"


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: load models
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transformer, shap_explainer, seg_bundle, full_pipeline

    print("\n" + "=" * 65)
    print("  Customer Churn API — Connecting to DagsHub")
    print("=" * 65)

    # 1. Initialize DagsHub Connection
    # It's best to use environment variables for the token in production/Docker
    dagshub.init(repo_owner="pritc2611", repo_name="Churn-models")

    download_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTERED_MODEL}/2",
        dst_path=str(MODEL_DIR)
    )

    model_uri = "models:/TelcoChurnModel/1" 
    full_pipeline = mlflow.sklearn.load_model(model_uri)
    print("✅  Main Pipeline loaded from DagsHub")

    # 3. Load the Clustering Model
    # If this is also registered, use its model_uri. 
    # If it's just an artifact in the same run, use mlflow.artifacts.download_artifacts
    seg_bundle    = joblib.load(MODEL_DIR / "KMeans-cluster-model.joblib")
    print("✅  Clustering model loaded")

    # 4. Set up components as before
    model = full_pipeline.named_steps["model"]
    transformer = full_pipeline.named_steps["transformation"]

    # Loading background data (ensure this file is in your Docker image or DVC)
    background = pd.read_csv(SHAPE_DIR / "shap_background.csv")
    shap_explainer = shap.Explainer(model.predict_proba, masker=background)
    print("✅  SHAP explainer ready")
    print("=" * 65 + "\n")

    yield
    print("Shutting down …")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-grade ML system for Telco customer churn prediction. "
        "Supports individual predictions (browser form or JSON/curl) "
        "and batch predictions (CSV / Excel upload → enriched Excel download)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_form():
    """Render the prediction input form."""
    return load_html_template("page.html")


# ── Individual prediction ─────────────────────────────────────────────────
@app.post("/predict", tags=["Prediction"])
async def predict(
    request: Request,
    gender: str           = Form(...),
    SeniorCitizen: str    = Form(...),
    Partner: str          = Form(...),
    Dependents: str       = Form(...),
    tenure: float         = Form(...),
    PhoneService: str     = Form(...),
    MultipleLines: str    = Form(...),
    InternetService: str  = Form(...),
    OnlineSecurity: str   = Form(...),
    OnlineBackup: str     = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str      = Form(...),
    StreamingTV: str      = Form(...),
    StreamingMovies: str  = Form(...),
    Contract: str         = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str    = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float   = Form(...),
):
    try:
        # ── Build customer input ─────────────────────────────
        customer = CustomerInput(
            gender=gender,
            SeniorCitizen=str(SeniorCitizen),
            Partner=Partner,
            Dependents=Dependents,
            tenure=tenure,
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges,
        )
        input_df = customer.to_dataframe()

        # ── Make prediction ────────────────────────────────
        X_transformed   = transformer.transform(input_df[ALL_FEATURES])
        churn_proba_raw = float(model.predict_proba(X_transformed)[0][1])
        churn_proba_pct = round(churn_proba_raw * 100, 1)
        risk_level      = get_risk_tier(churn_proba_raw)
        risk_color      = get_risk_color(churn_proba_raw)

        # ── SHAP explanations ─────────────────────────────
        shap_values = shap_explainer(X_transformed)
        reasons     = generate_reasons(shap_values)
        reasons_html = (
            '<ul style="list-style-type:none;padding-left:0;">'
            + "".join(f"<li>➡️ {r}</li>" for r in reasons)
            + "</ul>"
        )

        # ── Segmentation ─────────────────────────────────
        seg_row = input_df.copy()
        seg_row["ServiceCount"] = build_service_count(seg_row)
        segment = get_segment(seg_bundle, seg_row)

        # ── KPI calculations ─────────────────────────────
        kpis = calculate_individual_kpis(
            total_charges     = TotalCharges,
            monthly_charges   = MonthlyCharges,
            tenure            = tenure,
            churn_probability = churn_proba_raw,
            contract          = Contract,
        )

        # ── Render HTML template ─────────────────────────
        return templates.TemplateResponse(
            "prediction.html",
            {
                "request"          : request,
                "risk_level"       : risk_level,
                "risk_color"       : risk_color,
                "churn_probability": churn_proba_pct,
                "gender"           : gender,
                "SeniorCitizen"    : SeniorCitizen,
                "Partner"          : Partner,
                "Dependents"       : Dependents,
                "tenure"           : tenure,
                "PhoneService"     : PhoneService,
                "MultipleLines"    : MultipleLines,
                "InternetService"  : InternetService,
                "Contract"         : Contract,
                "PaperlessBilling" : PaperlessBilling,
                "PaymentMethod"    : PaymentMethod,
                "MonthlyCharges"   : MonthlyCharges,
                "TotalCharges"     : TotalCharges,
                "reasons_html"     : reasons_html,
                "segment"          : segment,
                **{f"kpi_{k}": v for k, v in kpis.items()},
            },
        )

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

# ── Individual prediction via JSON body ───────────────────────────────────
@app.post("/api/predict", tags=["Prediction"])
async def predict_json(customer: CustomerInput):
    """
    Pure JSON endpoint — use with requests / curl / Postman.

    Example:
        curl -X POST http://localhost:8000/api/predict \\
             -H "Content-Type: application/json" \\
             -d '{...}'
    """
    try:
        input_df = customer.to_dataframe()

        X_transformed   = transformer.transform(input_df[ALL_FEATURES])
        churn_proba_raw = float(model.predict_proba(X_transformed)[0][1])
        churn_proba_pct = round(churn_proba_raw * 100, 1)

        shap_values = shap_explainer(X_transformed)
        reasons     = generate_reasons(shap_values)

        seg_row = input_df.copy()
        seg_row["ServiceCount"] = build_service_count(seg_row)
        segment = get_segment(seg_bundle, seg_row)

        kpis = calculate_individual_kpis(
            total_charges     = customer.TotalCharges,
            monthly_charges   = customer.MonthlyCharges,
            tenure            = customer.tenure,
            churn_probability = churn_proba_raw,
            contract          = customer.Contract,
        )

        return {
            "status": "success",
            "data": {
                "churn_probability_pct": churn_proba_pct,
                "risk_level"           : get_risk_tier(churn_proba_raw),
                "segment"              : segment,
                "kpis"                 : kpis,
                "shap_factors"         : reasons,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Batch prediction ──────────────────────────────────────────────────────
@app.post("/batch_predict", tags=["Batch"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file with customer data.

    Returns an enriched Excel file (sorted by Revenue_At_Risk desc) with:
      - Churn_Probability, Risk_Tier
      - Segment, CLV, CLV_Tier, Revenue_At_Risk

    The file downloads automatically in the browser.
    """
    try:
        raw_bytes = await file.read()
        df        = read_uploaded_file(raw_bytes, file.filename)

        enriched_df = run_batch_prediction(
            df         = df,
            pipeline   = full_pipeline,
            seg_bundle = seg_bundle,
        )

        excel_bytes = df_to_excel_bytes(enriched_df)

        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": 'attachment; filename="churn_predictions.xlsx"'
            },
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Batch upload page ─────────────────────────────────────────────────────
@app.get("/batch", response_class=HTMLResponse, tags=["UI"])
def batch_page():
    """Render the batch prediction upload page."""
    return load_html_template("batch.html")


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Ops"])
def health_check():
    return {
        "status"            : "healthy",
        "model_loaded"      : model       is not None,
        "transformer_loaded": transformer is not None,
        "shap_loaded"       : shap_explainer is not None,
        "seg_loaded"        : seg_bundle  is not None,
    }