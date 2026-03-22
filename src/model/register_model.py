import os
import sys
import json
import pickle
import logging
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dagshub_config import setup_dagshub, set_experiment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.debug('Model loaded from %s', model_path)
    return model


def load_vectorizer(vectorizer_path: str):
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    logger.debug('Vectorizer loaded from %s', vectorizer_path)
    return vectorizer


def load_test_data() -> pd.DataFrame:
    test_path = os.path.join(PROJECT_ROOT, 'artifacts', 'interim', 'test_processed.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    df = pd.read_csv(test_path)
    df.fillna('', inplace=True)
    logger.debug('Test data loaded from %s', test_path)
    return df


def register_model_in_mlflow() -> bool:
    try:
        setup_dagshub()
        set_experiment('dvc-pipeline-runs')

        exp_info_path = os.path.join(PROJECT_ROOT, 'experiment_info.json')
        if not os.path.exists(exp_info_path):
            logger.error('experiment_info.json not found. Run model_evaluation first.')
            return False

        with open(exp_info_path) as f:
            experiment_info = json.load(f)

        run_id = experiment_info.get('run_id')
        if not run_id:
            logger.error('No run_id in experiment_info.json')
            return False

        logger.debug('Using run_id: %s', run_id)

        model_uri  = f"runs:/{run_id}/lgbm_model"
        model_name = "reddit_sentiment_lgbm"

        client = mlflow.tracking.MlflowClient()
        result = mlflow.register_model(model_uri, model_name)
        logger.info('Model registered as %s version %s', model_name, result.version)

        client.transition_model_version_stage(
            name=model_name, version=result.version, stage="Staging"
        )
        client.update_model_version(
            name=model_name, version=result.version,
            description=f"LightGBM sentiment model. Run ID: {run_id}",
        )

        # Validate with test data
        model_path      = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'lgbm_model.pkl')
        vectorizer_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'tfidf_vectorizer.pkl')
        model      = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)
        test_data  = load_test_data()

        X_test = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
            "recall":    recall_score(y_test, y_pred, average='weighted', zero_division=1),
            "f1":        f1_score(y_test, y_pred, average='weighted', zero_division=1),
        }

        logger.info('Validation metrics: %s', metrics)
        for name, value in metrics.items():
            client.log_metric(run_id, f"registered_{name}", value)

        reg_info = {
            "model_name":    model_name,
            "model_version": result.version,
            "run_id":        run_id,
            "stage":         "Staging",
            "metrics":       metrics,
            "timestamp":     pd.Timestamp.now().isoformat(),
        }
        reg_info_path = os.path.join(PROJECT_ROOT, 'model_registration_info.json')
        with open(reg_info_path, 'w') as f:
            json.dump(reg_info, f, indent=4)

        logger.debug('Registration info saved to %s', reg_info_path)
        return True

    except Exception as e:
        logger.error('Failed to register model: %s', e)
        return False


def main():
    logger.info('Starting model registration...')

    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'lgbm_model.pkl')
    exp_info   = os.path.join(PROJECT_ROOT, 'experiment_info.json')

    if not os.path.exists(model_path):
        logger.error('Model file not found. Run model_building first.')
        return
    if not os.path.exists(exp_info):
        logger.error('experiment_info.json not found. Run model_evaluation first.')
        return

    success = register_model_in_mlflow()
    if success:
        logger.info('Model registration complete!')
    else:
        logger.error('Model registration failed.')


if __name__ == "__main__":
    main()