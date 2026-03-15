import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

import wandb
from mlflow.client import MlflowClient

WANDB_PROJECT = "customer_churn_telco"
REGISTERED_MODEL = "TelcoChurnModel"
BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR.parent / "data" / "Telco-Customer-Churn.csv"
TRAINING_PATH = BASE_DIR.parent / "training" / "training.py"
REQUIRED_FILES = [
    BASE_DIR / "models" / "churn_clf.joblib",
    BASE_DIR / "models" / "KMeans-cluster-model.joblib",
    BASE_DIR / "shape-background" / "shap_background.csv",
]
models_dir = "./Churn-system/models"

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
        start_server()
    elif args.train_only:
        validate_data()
        run_training()
    else:
        validate_data()
        run_training()
        start_server()
