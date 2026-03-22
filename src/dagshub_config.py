import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
DAGSHUB_TOKEN    = os.getenv("DAGSHUB_TOKEN", "").strip()
REPO_NAME        = os.getenv("REPO_NAME", "reddit-sentiment-analysis").strip()

TRACKING_URI = f"https://dagshub.com/Rakesh-Kamath/reddit-sentiment-analysis.mlflow"


def setup_dagshub():
    if DAGSHUB_TOKEN:
        dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # This line properly initialises the DagsHub+MLflow connection
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def set_experiment(experiment_name: str = "dvc-pipeline-runs"):
    mlflow.set_experiment(experiment_name)