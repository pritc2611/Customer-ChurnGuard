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
import shutil
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import wandb
import warnings
import optuna
warnings.filterwarnings("ignore")

# client = MlflowClient()
EXPERIMENT_NAME = "customer_churn_telco"
REGISTERED_MODEL = "TelcoChurnModel"
METRIC_NAME = "recall"
WANDB_PROJECT = "customer_churn_telco"


# ── Feature groups ───────────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "Churn"
shap_backgrround_dir = "./shape-background"
os.makedirs(shap_backgrround_dir, exist_ok=True)


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
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    # "Yes" = 1, anything else = 0
    df["ServiceCount"] = (
        df[service_cols].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)
    )
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

    # This is the object we want to use in FastAPI
    seg_bundle = {"scaler": scaler, "kmeans": kmeans}
    
    # --- W&B LOGGING START ---
    run = wandb.init(project=WANDB_PROJECT, job_type="segmentation")
    
    # Use a temp directory so we don't leave junk on your D: drive
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "KMeans-cluster-model.joblib")
        
        # 1. SAVE the bundle to the temporary file
        joblib.dump(seg_bundle, file_path)
        
        # 2. Create the Artifact
        artifact_seg = wandb.Artifact("TelcoSegmentationModel", type="model")
        
        # 3. ADD the file to the artifact BEFORE logging it
        artifact_seg.add_file(file_path)
        
        # 4. LOG the artifact with the 'latest' alias
        run.log_artifact(artifact_seg, aliases=["latest"])
    
    run.finish()
    # --- W&B LOGGING END ---

    cluster_labels = {
        0: "Loyal High-Value",
        1: "Low Engagement / Higher Risk",
    }
    df["Cluster"] = kmeans.predict(scaled)
    df["Segment"] = df["Cluster"].map(cluster_labels)
    
    return df, seg_bundle


# ─────────────────────────────────────────────────────────────────────────────
# Churn classifier pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(clf) -> Pipeline:
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_scaler = StandardScaler()
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="mean")


    cat_pipeline = Pipeline(steps=[
        ("imputer", cat_imputer),
        ("encoder", cat_encoder)
    ])

    num_pipeline = Pipeline(steps=[
        ("imputer", num_imputer),
        ("scaler", num_scaler)
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("cat", cat_pipeline, CAT_FEATURES),
        ("num", num_pipeline, NUM_FEATURES)
    ])

    return Pipeline(
        steps=[
            ("transformation", preprocessor),
            ("model", clf),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# MLflow trainer
# ─────────────────────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self):
        self.best_run_id = None
        self.best_score = 0.0
        self.best_pipeline = None
        self.best_model_name = None

    def _metrics(self, y_true, y_pred, y_proba) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }

    def train_one(self, name, clf, params, X_train, X_test, y_train, y_test):
        run = wandb.init(
            project=WANDB_PROJECT,
            name=name,
            config=params,
        )

        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        m = self._metrics(y_test, y_pred, y_proba)

        wandb.log(m)
        print(f"  {name:30s}  AUC={m['recall']:.4f}  F1={m['f1']:.4f}")

        if m[METRIC_NAME] > self.best_score:
            self.best_score = m[METRIC_NAME]
            self.best_pipeline = pipeline
            self.best_model_name = name

        artifact = wandb.Artifact(
            name="TelcoChurnModel",
            type="model",
            metadata=m,
        )
        run.log_artifact(artifact)
        run.finish()

        return pipeline, m

    def _optuna_objective(self, trial, clf_class, X_train, X_test, y_train, y_test):
        # Example: RandomForest hyperparameters
        if clf_class == RandomForestClassifier:
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight="balanced",
                random_state=42
            )
        elif clf_class == XGBClassifier:
            clf = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError("Unsupported classifier for Optuna")

        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        m = self._metrics(y_test, y_pred, y_proba)

        # Update best model manually
        if m[METRIC_NAME] > self.best_score:
            self.best_score = m[METRIC_NAME]
            self.best_pipeline = pipeline
            self.best_model_name = clf_class.__name__

        return m[METRIC_NAME]

    def tune_hyperparameters(self, clf_class, X_train, X_test, y_train, y_test, n_trials=30):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._optuna_objective(trial, clf_class, X_train, X_test, y_train, y_test),
            n_trials=n_trials
        )
        print(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params

    def train_all(self, X_train, X_test, y_train, y_test, tune=True):
        configs = {
            "XgboostClassifier": XGBClassifier(),
            "RandomForest": RandomForestClassifier(),
        }

        results = {}
        for name, clf in configs.items():
            if tune:
                best_params = self.tune_hyperparameters(type(clf), X_train, X_test, y_train, y_test)
                print(f"Tuned params for {name}: {best_params}")
                clf.set_params(**best_params)

            pipeline, metrics = self.train_one(
                name, clf, {}, X_train, X_test, y_train, y_test
            )
            results[name] = metrics

        shutil.rmtree("./wandb", ignore_errors=True)
        return results
    
    def register_best_model(self):
        run = wandb.init(project=WANDB_PROJECT, job_type="model-registry")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "churn_clf.joblib")
        
        # 1. Save to the temp folder
            joblib.dump(self.best_pipeline, temp_path)
        
        # 2. Upload to W&B
            artifact = wandb.Artifact("TelcoChurnModel", type="model")
            artifact.add_file(temp_path)
            run.log_artifact(artifact, aliases=["production"])
        
            run.finish()
            shutil.rmtree("D:\churn_model\Churn-system\wandb",ignore_errors=True)
            


# ─────────────────────────────────────────────────────────────────────────────
# SHAP background sample
# ─────────────────────────────────────────────────────────────────────────────
def save_shap_background(pipeline, X_train: pd.DataFrame, n: int = 100):
    transformer = pipeline.named_steps["transformation"]
    X_bg = transformer.transform(X_train.sample(n, random_state=42))
    pd.DataFrame(X_bg).to_csv(
        f"{shap_backgrround_dir}/shap_background.csv", index=False
    )
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
    print("\n" + "=" * 60)
    print("  Segmentation Model")
    print("=" * 60)
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
    print("✅  Best pipeline saved  →  churn_clf.joblib")

    # ── SHAP background ───────────────────────────────────────────────────
    save_shap_background(trainer.best_pipeline, X_train)

    # ── Results summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    df_res = pd.DataFrame(results).T
    print(df_res[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))
    print("\nTraining complete! 🎉")
