import logging
import os
import pickle
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple):
    try:
        models_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
        os.makedirs(models_dir, exist_ok=True)

        text_column = 'clean_comment' if 'clean_comment' in train_data.columns else 'processed_text'

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(train_data[text_column])

        label_column = 'category' if 'category' in train_data.columns else 'label'
        y_train = train_data[label_column]

        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        logger.debug('Vectorizer saved to %s', vectorizer_path)

        return X_train, y_train
    except Exception as e:
        logger.error('Failed to apply TF-IDF: %s', e)
        raise


def train_lgbm(X_train, y_train, learning_rate: float, max_depth: int, n_estimators: int):
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
        )
        model.fit(X_train, y_train)
        logger.debug('LightGBM model trained successfully')
        return model
    except Exception as e:
        logger.error('Failed to train LightGBM: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Failed to save model: %s', e)
        raise


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        full_path = os.path.join(PROJECT_ROOT, params_path)
        with open(full_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters loaded from %s', full_path)
        return params
    except FileNotFoundError:
        logger.error('params.yaml not found at %s', full_path)
        raise


def main():
    try:
        params = load_params('params.yaml')

        max_features  = params['model_building']['max_features']
        ngram_range   = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth     = params['model_building']['max_depth']
        n_estimators  = params['model_building']['n_estimators']

        logger.debug('Parameters: max_features=%s, ngram_range=%s', max_features, ngram_range)

        train_path = os.path.join(PROJECT_ROOT, 'artifacts', 'interim', 'train_processed.csv')
        train_data = load_data(train_path)

        X_train, y_train = apply_tfidf(train_data, max_features, ngram_range)
        model = train_lgbm(X_train, y_train, learning_rate, max_depth, n_estimators)

        model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'lgbm_model.pkl')
        save_model(model, model_path)
        logger.debug('Model saved to %s', model_path)

    except KeyError as e:
        logger.error('Missing key in params.yaml: %s', e)
        raise


if __name__ == "__main__":
    main()