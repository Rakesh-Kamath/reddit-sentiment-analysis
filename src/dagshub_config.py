import os
from dotenv import load_dotenv
import dagshub
import mlflow

# -------------------------------------------------
# Initialize DagsHub + MLflow (AUTH FIRST)
# -------------------------------------------------
def setup_dagshub():
    load_dotenv()

    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
    REPO_NAME = os.getenv("REPO_NAME", "reddit-sentiment-analysis").strip()

    if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
        raise RuntimeError("❌ DagsHub credentials not found in environment variables")

    # 🔐 Authenticate with DagsHub FIRST
    dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)

    # Set MLflow environment variables
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    print(f"✅ DagsHub/MLflow initialized")
    print(f"🔗 Tracking URI: {tracking_uri}")

    return mlflow

# -------------------------------------------------
# Set / create MLflow experiment (SAFE)
# -------------------------------------------------
def set_experiment(experiment_name="dvc-pipeline-runs"):
    setup_dagshub()   # 🔥 always initialize first
    mlflow.set_experiment(experiment_name)
    print(f"✅ Experiment set: {experiment_name}")