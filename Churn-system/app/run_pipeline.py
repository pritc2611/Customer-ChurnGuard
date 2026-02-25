import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient

os.environ["MLFLOW_TRACKING_USERNAME"] = "pritc2611"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d69891a2caee7f83dd7ff7cea972fc432996f030"

mlflow.set_tracking_uri("https://dagshub.com/pritc2611/Churn-models.mlflow")
print(mlflow.get_tracking_uri())

REGISTERED_MODEL = "TelcoChurnModel"
BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR.parent / "data" / "Telco-Customer-Churn.csv"
TRAINING_PATH = BASE_DIR.parent / "training" / "training.py"
REQUIRED_FILES = [
    BASE_DIR / "models" / "churn_clf.joblib",
    BASE_DIR / "models" / "KMeans-cluster-model.joblib",
    BASE_DIR / "shape-background" / "shap_background.csv",
]
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)


def banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
def validate_data():
    banner("Step 1 — Data Validation")
    if not Path(DATA_PATH).exists():
        print(f"❌  Dataset not found: {DATA_PATH}")
        print("    Place your CSV at ./data/Telco-Customer-Churn.csv and retry.")
        sys.exit(1)

    import pandas as pd

    df = pd.read_csv(DATA_PATH, nrows=5)
    required_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌  Missing columns: {missing}")
        sys.exit(1)
    print(f"✅  Dataset OK  →  {DATA_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
def run_training():
    banner("Step 2 — Model Training")
    result = subprocess.run([sys.executable, TRAINING_PATH], check=False)
    if result.returncode != 0:
        print("❌  Training failed.")
        sys.exit(1)
    print("✅  Training complete")


# ─────────────────────────────────────────────────────────────────────────────
def download_model_from_registry():
    """
    Pull Production model from MLflow Registry (DagsHub)
    and store it in ./models/churn_clf.joblib
    """
    banner("Step 3 — Download Best Model from MLflow Registry")

    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
    except Exception as e:
        print(f"❌  Cannot reach model registry: {e}")
        sys.exit(1)

    if not prod_versions:
        print("⚠️  No Production model found in registry.")
        return

    # Pick latest Production version
    best = sorted(prod_versions, key=lambda v: int(v.version), reverse=True)[0]

    print(f"Model: {REGISTERED_MODEL} v{best.version} (run: {best.run_id})")

    # Download model artifact
    artifact_path = client.download_artifacts(
        run_id=best.run_id, path="", dst_path="./tmp_model"
    )
    src = Path(artifact_path) / "model.pkl"
    if src.exists():
        import joblib

        joblib.dump(joblib.load(src), Path(models_dir) / "churn_clf.joblib")
        print(f"✅  churn_clf.joblib updated from registry (v{best.version})")
    else:
        print("⚠️  model.pkl not found — keeping existing churn_clf.joblib")

        shutil.rmtree("./tmp_model", ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
def check_artifacts():
    banner("Step 4 — Artifact Check")
    for f in REQUIRED_FILES:
        if Path(f).exists():
            print(f"  ✅  {f}")
        else:
            print(f"  ❌  {f}  — MISSING")
            sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def start_server():
    banner("Step 5 — Starting FastAPI Server")
    print("  URL: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")
    subprocess.run(
        ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        check=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn MLOps Pipeline")
    parser.add_argument(
        "--serve-only", action="store_true", help="Skip training, start API only"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Train only, no API server"
    )
    args = parser.parse_args()

    if args.serve_only:
        check_artifacts()
        start_server()
    elif args.train_only:
        validate_data()
        run_training()
        download_model_from_registry()
    else:
        validate_data()
        run_training()
        download_model_from_registry()
        check_artifacts()
        start_server()
