import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dagshub_config import setup_dagshub, set_experiment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug('Data loaded from %s  shape=%s', file_path, df.shape)
    return df


def load_model(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model not found: {file_path}")
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    logger.debug('Model loaded from %s', file_path)
    return model


def load_vectorizer(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vectorizer not found: {file_path}")
    vectorizer = joblib.load(file_path)
    logger.debug('Vectorizer loaded from %s', file_path)
    return vectorizer


def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    logger.debug('Test accuracy: %.4f', report.get('accuracy', 0))
    return report, cm


def log_confusion_matrix(cm, dataset_name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix – {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    logger.debug('Confusion matrix saved to %s', cm_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    info = {
        'run_id': run_id,
        'model_path': model_path,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=4)
    logger.debug('Model info saved to %s', file_path)


def main():
    try:
        setup_dagshub()
        set_experiment('dvc-pipeline-runs')

        with mlflow.start_run() as run:
            import yaml
            params_path = os.path.join(PROJECT_ROOT, 'params.yaml')
            with open(params_path) as f:
                params = yaml.safe_load(f)
            for key, value in params.items():
                mlflow.log_param(key, str(value))

            model_path      = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'lgbm_model.pkl')
            vectorizer_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'tfidf_vectorizer.pkl')
            test_path       = os.path.join(PROJECT_ROOT, 'artifacts', 'interim', 'test_processed.csv')

            model      = load_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)
            test_data  = load_data(test_path)

            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            input_example = pd.DataFrame(
                X_test[:5].toarray(),
                columns=vectorizer.get_feature_names_out()
            )
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            mlflow.sklearn.log_model(
                model, "lgbm_model",
                signature=signature,
                input_example=input_example,
            )

            save_model_info(run.info.run_id, "lgbm_model",
                            os.path.join(PROJECT_ROOT, 'experiment_info.json'))

            if os.path.exists(vectorizer_path):
                mlflow.log_artifact(vectorizer_path)

            report, cm = evaluate_model(model, X_test, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    try:
                        mlflow.log_metrics({
                            f"test_{label}_precision": metrics.get('precision', 0),
                            f"test_{label}_recall":    metrics.get('recall', 0),
                            f"test_{label}_f1":        metrics.get('f1-score', 0),
                        })
                    except Exception:
                        pass

            if 'accuracy' in report:
                mlflow.log_metric("test_accuracy", report['accuracy'])

            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tags({
                "model_type": "LightGBM",
                "task": "Sentiment Analysis",
                "dataset": "Reddit Comments",
            })

            logger.info('Model evaluation complete.  Run ID: %s', run.info.run_id)
            logger.info('Test accuracy: %.4f', report.get('accuracy', 0))

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        print(f"\nError: {e}")
        print("Tip: run 'dvc repro' to generate all required files first.")
        raise
    except Exception as e:
        logger.error('Evaluation failed: %s', e)
        raise


if __name__ == '__main__':
    main()