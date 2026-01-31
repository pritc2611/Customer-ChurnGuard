from fastapi import FastAPI, Form, HTTPException , Request 
from fastapi.responses import HTMLResponse , JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from function import customerinput , generate_reasons
from fastapi.templating import Jinja2Templates
from function import load_html_template 
import joblib
import shap
from contextlib import asynccontextmanager



@asynccontextmanager
async def lifespan(app:FastAPI):
    global model , transformer , background , shap_explainer

    print("\n" + "="*70)
    print("Starting Startup")
    print("\n" + "="*70)

    print("Loading pipelines")
    print("\n" + "="*70)

    full_pipline = joblib.load("churn_clf.joblib")
    print("Pipelines successfully loaded...")

    model = full_pipline.named_steps["model"]
    transformer = full_pipline.named_steps["trf"]

    print("\n" + "="*70)
    background = pd.read_csv("shap_background.csv")
    print(" successfully loaded background files")
    print("\n" + "="*70)

    print("\n" + "="*70)
    shap_explainer = shap.Explainer(model.predict_proba, masker=background)
    print("SHAP successfully loaded")
    print("\n" + "="*70)

    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-grade ML system for predicting customer churn",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
template = Jinja2Templates(directory="template")


@app.get("/", response_class=HTMLResponse)
def form():
    """Render the prediction form"""
    return load_html_template("page.html")


@app.post("/predict")
async def predict(request: Request,
    Gender: str = Form(...),
    Subscription_Type: str = Form(...),
    Contract_Length: str = Form(...),
    Tenure: float = Form(...),
    Age: int = Form(...),
    Usage_Frequency: float = Form(...),
    Total_Spend: float = Form(...)
):
    print("trying to predict")
    try:
        customer_data = customerinput(
            gender=Gender,
            Subscription_type=Subscription_Type,
            contract_length=Contract_Length,
            tenure=Tenure,
            Age=Age,
            Usage_Frequency=Usage_Frequency,
            Total_spend=Total_Spend
        )
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            'Gender': customer_data.gender,
            'Subscription Type': customer_data.Subscription_type,
            'Contract Length': customer_data.contract_length,
            'Tenure': customer_data.tenure,
            'Age': customer_data.Age,
            'Usage Frequency': customer_data.Usage_Frequency,
            'Total Spend': customer_data.Total_spend
        }])
        print("created dataframe")
        
        # Transform data and make prediction
        transform_data = transformer.transform(input_df)
        prediction = model.predict_proba(transform_data)
        churn_probability = prediction[0][1] * 100
        print(churn_probability)
        print("prediction done")
        
        # Get SHAP values
        print("explaining")
        shap_values = shap_explainer(transform_data)
        print("explaining done")

        reasons = generate_reasons(shap_val=shap_values)
        print("reasons genrated")

        # Determine risk level
        if churn_probability >= 70:
            risk_level = "High Risk 🚨"
        elif churn_probability >= 40:
            risk_level = "Medium Risk ⚠️"
        else:
            risk_level = "Low Risk ✅"

        reasons_html = "<ul>" + "".join(f"<li>➡️ {s}</li>" for s in reasons) + "</ul>"

        print("returning probability")
        return template.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "risk_level": risk_level,
                "churn_probability": round(churn_probability, 1),
                "Gender": Gender,
                "Age": Age,
                "Subscription_Type": Subscription_Type,
                "Contract_Length": Contract_Length,
                "Tenure": Tenure,
                "Usage_Frequency": Usage_Frequency,
                "Total_Spend": Total_Spend,
                "reasons_html":reasons_html
            }
        )
            

    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/api/health")
def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "transformer_loaded": transformer is not None,
        "shap_loaded": shap_explainer is not None
    }


@app.post("/api/predict")
async def api_predict(customer: customerinput):
    """JSON API endpoint for predictions"""
    try:
        input_df = pd.DataFrame([{
            'Gender': customer.Gender,
            'Subscription Type': customer.Subscription_Type,
            'Contract Length': customer.Contract_Length,
            'Tenure': customer.Tenure,
            'Age': customer.Age,
            'Usage Frequency': customer.Usage_Frequency,
            'Total Spend': customer.Total_Spend
        }])
        
        # Transform and predict
        transform_data = transformer.transform(input_df)
        prediction = model.predict_proba(transform_data)
        
        # Get SHAP values and reasons
        shap_values = shap_explainer(transform_data)
        reasons = generate_reasons(shap_val=shap_values)
        
        return {
            "churn_prediction": int(prediction[0][1] > 0.5),
            "churn_probability": float(prediction[0][1]),
            "risk_level": "High" if prediction[0][1] >= 0.7 else "Medium" if prediction[0][1] >= 0.4 else "Low",
            "reasons": reasons,
            "input_data": customer.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))