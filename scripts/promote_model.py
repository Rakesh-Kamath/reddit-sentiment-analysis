import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dagshub_config import setup_dagshub, set_experiment

# Authenticate with DagsHub
dagshub.auth.add_app_token(token=os.environ.get("DAGSHUB_TOKEN"))
setup_dagshub()
set_experiment()

client = MlflowClient()

MODEL_NAME = "reddit_sentiment_lgbm"

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    raise RuntimeError(f"❌ No versions found for model '{MODEL_NAME}'. Did you register it?")

latest_version = max(int(v.version) for v in versions)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ Model '{MODEL_NAME}' version {latest_version} promoted to Production")