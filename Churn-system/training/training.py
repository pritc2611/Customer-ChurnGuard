"""
Customer Churn Prediction - Training Pipeline
Trains churn classifier + KMeans segmentation model.
Tracks experiments with MLflow and registers the best model.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import dagshub

import warnings
warnings.filterwarnings("ignore")

dagshub.init(repo_owner="pritc2611", repo_name="Churn-models", mlflow=True)
print(mlflow.get_tracking_uri())

client = MlflowClient()
EXPERIMENT_NAME   = "customer_churn_telco"
REGISTERED_MODEL  = "TelcoChurnModel"
METRIC_NAME       = "recall"

# ── Feature groups ───────────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET       = "Churn"
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True) 
shap_backgrround_dir = "./shape-background"
os.makedirs(shap_backgrround_dir,exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Drop customer ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges can come in as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Encode target
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    
    df[TARGET] = df[TARGET].astype(int)

    # Cast SeniorCitizen to str so it's treated as categorical
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    print(f"Clean shape: {df.shape}  |  Churn rate: {df[TARGET].mean():.2%}")
    return df


def build_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count value-added services per customer (used for segmentation)."""
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # "Yes" = 1, anything else = 0
    df["ServiceCount"] = df[service_cols].apply(
        lambda col: (col == "Yes").astype(int)
    ).sum(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation model
# ─────────────────────────────────────────────────────────────────────────────
def train_segmentation_model(df: pd.DataFrame):
    """Train KMeans (k=2) on tenure, MonthlyCharges, TotalCharges, ServiceCount."""
    seg_features = ["tenure", "MonthlyCharges", "TotalCharges", "ServiceCount"]
    seg_df = df[seg_features].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(seg_df)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(scaled)

    seg_bundle = {"scaler": scaler, "kmeans": kmeans}
    local_path = f"{models_dir}/KMeans-cluster-model.joblib"
    joblib.dump(seg_bundle, local_path)
    print(f"✅  Segmentation model saved locally  →  {local_path}")

    if mlflow.active_run():
        # This uploads the local file to the 'model' folder on DagsHub
        mlflow.log_artifact(local_path, artifact_path="model")
        print("🚀  Segmentation model uploaded to DagsHub MLflow!")

    cluster_labels = {
        0: "Loyal High-Value",
        1: "Low Engagement / Higher Risk",
    }
    df["Cluster"]  = kmeans.predict(scaled)
    df["Segment"]  = df["Cluster"].map(cluster_labels)
    return df, seg_bundle


# ─────────────────────────────────────────────────────────────────────────────
# Churn classifier pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(clf) -> Pipeline:
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_scaler  = StandardScaler()

    preprocessor = ColumnTransformer(transformers=[
        ("cat", cat_encoder, CAT_FEATURES),
        ("num", num_scaler,  NUM_FEATURES),
    ])

    return Pipeline(steps=[
        ("transformation", preprocessor),
        ("model",          clf),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MLflow trainer
# ─────────────────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self):
        if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
            client.create_experiment(EXPERIMENT_NAME)
        self.best_run_id   = None
        self.best_score    = 0.0
        self.best_pipeline = None

    def _metrics(self, y_true, y_pred, y_proba) -> dict:
        return {
            "accuracy" : accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall"   : recall_score(y_true, y_pred, zero_division=0),
            "f1"       : f1_score(y_true, y_pred, zero_division=0),
            "roc_auc"  : roc_auc_score(y_true, y_proba),
        }

    def train_one(self, name, clf, params, X_train, X_test, y_train, y_test):
        pipeline = build_pipeline(clf)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            pipeline.fit(X_train, y_train)

            y_pred  = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            m = self._metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"  {name:30s}  AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")

            if m[METRIC_NAME] > self.best_score:
                self.best_score    = m[METRIC_NAME]
                self.best_run_id   = run.info.run_id
                self.best_pipeline = pipeline

        return pipeline, m

    def train_all(self, X_train, X_test, y_train, y_test):
        configs = {
            "XgboostClassifier": (
                XGBClassifier(),
                {"nothing":0},
            ),
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, max_depth=8,
                    min_samples_split=5, class_weight="balanced", random_state=42
                ),
                {"n_estimators": 100, "max_depth": 5},
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1,
                    max_depth=5, random_state=42
                ),
                {"n_estimators": 100, "lr": 0.05},
            ),
        }

        print("\n" + "="*60)
        print("  Model Training")
        print("="*60)
        results = {}
        for name, (clf, params) in configs.items():
            pipeline, metrics = self.train_one(
                name, clf, params, X_train, X_test, y_train, y_test
            )
            results[name] = metrics
        return results

    def register_best_model(self):
        model_uri = f"runs:/{self.best_run_id}/model"
        try:
            client.get_registered_model(REGISTERED_MODEL)
        except Exception:
            client.create_registered_model(REGISTERED_MODEL)
        mv = client.create_model_version(
            name=REGISTERED_MODEL,
            source=model_uri,
            run_id=self.best_run_id,)
        client.transition_model_version_stage(
                name=REGISTERED_MODEL,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True,)

        print(
            f"\n✅  Best model (recall={self.best_score:.4f}) "
            f"registered as '{REGISTERED_MODEL}' v{mv.version} → Production")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP background sample
# ─────────────────────────────────────────────────────────────────────────────
def save_shap_background(pipeline, X_train: pd.DataFrame, n: int = 100):
    transformer = pipeline.named_steps["transformation"]
    X_bg = transformer.transform(X_train.sample(n, random_state=42))
    pd.DataFrame(X_bg).to_csv(f"{shap_backgrround_dir}/shap_background.csv", index=False)
    print(f"✅  SHAP background ({n} rows) saved  →  shap_background.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_PATH = BASE_DIR.parent / "data" / "Telco-Customer-Churn.csv"

    # ── Load & clean ──────────────────────────────────────────────────────
    df = load_and_clean(DATA_PATH)
    df = build_service_count(df)

    # ── Segmentation ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Segmentation Model")
    print("="*60)
    df, seg_bundle = train_segmentation_model(df)

    # ── Train/test split ──────────────────────────────────────────────────
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain={len(X_train):,}  |  Test={len(X_test):,}")

    # ── Train classifiers ─────────────────────────────────────────────────
    trainer = ModelTrainer()
    results = trainer.train_all(X_train, X_test, y_train, y_test)

    # ── Register best model ───────────────────────────────────────────────
    trainer.register_best_model()

    # ── Save pipeline locally for API ────────────────────────────────────
    joblib.dump(trainer.best_pipeline, f"{models_dir}/churn_clf.joblib")
    print("✅  Best pipeline saved  →  churn_clf.joblib")

    # ── SHAP background ───────────────────────────────────────────────────
    save_shap_background(trainer.best_pipeline, X_train)

    # ── Results summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Results Summary")
    print("="*60)
    df_res = pd.DataFrame(results).T
    print(df_res[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))
    print("\nTraining complete! 🎉")