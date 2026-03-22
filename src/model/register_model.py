import os
import json
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dagshub_config import setup_dagshub, set_experiment

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path):
    """Load the trained model."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


def load_vectorizer(vectorizer_path):
    """Load the trained vectorizer."""
    try:
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

        vectorizer = joblib.load(vectorizer_path)
        logger.debug(f"✅ Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"❌ Error loading vectorizer: {e}")
        raise


def load_test_data():
    """Load test data for model validation."""
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        possible_paths = [
            os.path.join(root_dir, 'artifacts', 'interim', 'test_processed.csv'),
            'artifacts/data/interim/test_processed.csv',
            'artifacts/interim/test_processed.csv',
            'data/processed/test_processed.csv'
        ]

        test_data = None
        for path in possible_paths:
            if os.path.exists(path):
                test_data = pd.read_csv(path)
                test_data.fillna('', inplace=True)
                logger.debug(f"✅ Test data loaded from {path}")
                break

        if test_data is None:
            raise FileNotFoundError("Could not find test data in any expected location")

        return test_data
    except Exception as e:
        logger.error(f"❌ Error loading test data: {e}")
        raise


def register_model_in_mlflow():
    """Register the model in MLflow model registry."""
    try:
        setup_dagshub()
        set_experiment('dvc-pipeline-runs')

        if not os.path.exists('experiment_info.json'):
            logger.error("experiment_info.json not found. Run model_evaluation first.")
            return False

        with open('experiment_info.json', 'r') as f:
            experiment_info = json.load(f)

        run_id = experiment_info.get('run_id')
        if not run_id:
            logger.error("No run_id found in experiment_info.json")
            return False

        logger.debug(f"Found run_id: {run_id}")

        client = mlflow.tracking.MlflowClient()

        model_uri = f"runs:/{run_id}/lgbm_model"
        model_name = "reddit_sentiment_lgbm"
        result = mlflow.register_model(model_uri, model_name)

        logger.info(f"✅ Model registered as: {model_name} (version {result.version})")

        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )
        logger.debug("Model moved to Staging stage")

        client.update_model_version(
            name=model_name,
            version=result.version,
            description=f"LightGBM model for Reddit sentiment analysis. Run ID: {run_id}"
        )

        test_data = load_test_data()

        model = load_model(os.path.join('artifacts', 'models', 'lgbm_model.pkl'))
        vectorizer = load_vectorizer(os.path.join('artifacts', 'models', 'tfidf_vectorizer.pkl'))

        X_test = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        logger.info("\n📊 Model Validation Metrics:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1 Score:  {metrics['f1']:.4f}")

        for metric_name, metric_value in metrics.items():
            client.log_metric(run_id, f"registered_{metric_name}", metric_value)

        registration_info = {
            "model_name": model_name,
            "model_version": result.version,
            "run_id": run_id,
            "stage": "Staging",
            "metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        with open('model_registration_info.json', 'w') as f:
            json.dump(registration_info, f, indent=4)

        logger.debug("Registration info saved to model_registration_info.json")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        return False


def main():
    """Main function to register the model."""
    try:
        logger.info("🚀 Starting model registration process...")

        if not os.path.exists(os.path.join('artifacts', 'models', 'lgbm_model.pkl')):
            logger.error("Model file not found. Run model_building first.")
            return

        if not os.path.exists('experiment_info.json'):
            logger.error("experiment_info.json not found. Run model_evaluation first.")
            return

        success = register_model_in_mlflow()

        if success:
            logger.info("✅ Model registration completed successfully!")
            logger.info("📝 Model registered in MLflow Model Registry")
            logger.info("🔗 View at: https://dagshub.com/Rakesh-Kamath/reddit-sentiment-analysis.mlflow")
        else:
            logger.error("❌ Model registration failed")

    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        raise


if __name__ == "__main__":
    main()