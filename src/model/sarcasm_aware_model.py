import os
import sys
import logging
import pickle
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sarcasm.sarcasm_detector import predict_sarcasm_probability, load_sarcasm_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('sarcasm_aware_model')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    logger.debug('Data loaded from %s  shape=%s', file_path, df.shape)
    return df


def load_params() -> dict:
    params_path = os.path.join(PROJECT_ROOT, 'params.yaml')
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def build_feature_matrix(texts: list, tfidf_vectorizer, sarcasm_model, sarcasm_vectorizer):
    logger.debug('Building TF-IDF features...')
    X_tfidf = tfidf_vectorizer.transform(texts)

    logger.debug('Computing sarcasm probabilities...')
    sarcasm_probs = predict_sarcasm_probability(texts, sarcasm_model, sarcasm_vectorizer)

    sarcasm_prob_col       = sarcasm_probs.reshape(-1, 1)
    sarcasm_binary_col     = (sarcasm_probs > 0.5).astype(float).reshape(-1, 1)
    sarcasm_confidence_col = np.abs(sarcasm_probs - 0.5).reshape(-1, 1)

    sarcasm_features = sp.csr_matrix(
        np.hstack([sarcasm_prob_col, sarcasm_binary_col, sarcasm_confidence_col])
    )
    X_combined = sp.hstack([X_tfidf, sarcasm_features])

    logger.debug('Combined feature matrix shape: %s (TF-IDF + 3 sarcasm features)', X_combined.shape)
    return X_combined


def train_baseline_model(X_train, y_train, params):
    logger.info('Training BASELINE model (TF-IDF only)...')
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        is_unbalance=True,
        class_weight='balanced',
        learning_rate=params['model_building']['learning_rate'],
        max_depth=params['model_building']['max_depth'],
        n_estimators=params['model_building']['n_estimators'],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_sarcasm_aware_model(X_train_combined, y_train):
    logger.info('Training SARCASM-AWARE model (TF-IDF + 3 sarcasm features)...')
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        is_unbalance=True,
        class_weight='balanced',
        learning_rate=0.05,
        max_depth=-1,
        n_estimators=500,
        num_leaves=63,
        min_child_samples=20,
        n_jobs=-1,
    )
    model.fit(X_train_combined, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    report    = classification_report(y_test, y_pred)

    logger.info('\n' + '='*60)
    logger.info('📊 %s Results:', model_name)
    logger.info('   Accuracy:  %.4f', accuracy)
    logger.info('   Precision: %.4f', precision)
    logger.info('   Recall:    %.4f', recall)
    logger.info('   F1 Score:  %.4f', f1)
    logger.info('\n%s', report)

    return {
        'model_name': model_name,
        'accuracy':   accuracy,
        'precision':  precision,
        'recall':     recall,
        'f1':         f1,
    }


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.debug('Model saved to %s', path)


def main():
    logger.info('🚀 Starting sarcasm-aware sentiment model training...')

    params = load_params()

    train_path = os.path.join(PROJECT_ROOT, 'artifacts', 'interim', 'train_processed.csv')
    test_path  = os.path.join(PROJECT_ROOT, 'artifacts', 'interim', 'test_processed.csv')

    train_df = load_data(train_path)
    test_df  = load_data(test_path)

    train_texts = train_df['clean_comment'].tolist()
    test_texts  = test_df['clean_comment'].tolist()
    y_train     = train_df['category'].values
    y_test      = test_df['category'].values

    vectorizer_path  = os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(vectorizer_path)
    logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)

    sarcasm_model, sarcasm_vectorizer = load_sarcasm_model()
    logger.debug('Sarcasm model loaded')

    X_train_tfidf    = tfidf_vectorizer.transform(train_texts)
    X_test_tfidf     = tfidf_vectorizer.transform(test_texts)

    X_train_combined = build_feature_matrix(train_texts, tfidf_vectorizer, sarcasm_model, sarcasm_vectorizer)
    X_test_combined  = build_feature_matrix(test_texts,  tfidf_vectorizer, sarcasm_model, sarcasm_vectorizer)

    baseline_model      = train_baseline_model(X_train_tfidf, y_train, params)
    sarcasm_aware_model = train_sarcasm_aware_model(X_train_combined, y_train)

    baseline_results      = evaluate_model(baseline_model,      X_test_tfidf,    y_test, 'Baseline (TF-IDF only)')
    sarcasm_aware_results = evaluate_model(sarcasm_aware_model, X_test_combined, y_test, 'Sarcasm-Aware (TF-IDF + 3 Sarcasm Features)')

    improvement_acc = (sarcasm_aware_results['accuracy'] - baseline_results['accuracy']) * 100
    improvement_f1  = (sarcasm_aware_results['f1']       - baseline_results['f1'])       * 100

    logger.info('\n' + '='*60)
    logger.info('📈 COMPARISON SUMMARY:')
    logger.info('   %-42s  Accuracy   F1 Score', 'Model')
    logger.info('   %-42s  --------   --------', '-'*42)
    logger.info('   %-42s  %.4f     %.4f', baseline_results['model_name'],      baseline_results['accuracy'],      baseline_results['f1'])
    logger.info('   %-42s  %.4f     %.4f', sarcasm_aware_results['model_name'], sarcasm_aware_results['accuracy'], sarcasm_aware_results['f1'])
    logger.info('   Accuracy Improvement: %+.2f%%', improvement_acc)
    logger.info('   F1 Improvement:       %+.2f%%', improvement_f1)
    logger.info('='*60)

    save_model(baseline_model,
               os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'baseline_lgbm.pkl'))
    save_model(sarcasm_aware_model,
               os.path.join(PROJECT_ROOT, 'artifacts', 'models', 'sarcasm_aware_lgbm.pkl'))

    logger.info('✅ Both models saved successfully!')

    results_df   = pd.DataFrame([baseline_results, sarcasm_aware_results])
    results_path = os.path.join(PROJECT_ROOT, 'artifacts', 'results_comparison.csv')
    results_df.to_csv(results_path, index=False)
    logger.info('📊 Results saved to %s', results_path)


if __name__ == '__main__':
    main()