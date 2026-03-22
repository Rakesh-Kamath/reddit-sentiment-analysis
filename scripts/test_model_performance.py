import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pytest
from dotenv import load_dotenv
import pickle

load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("REPO_NAME", "reddit-sentiment-analysis")

@pytest.mark.parametrize("model_path, holdout_data_path, vectorizer_path", [
    ("artifacts/models/lgbm_model.pkl", "artifacts/interim/test_processed.csv", "artifacts/models/tfidf_vectorizer.pkl"),
])
def test_model_performance(model_path, holdout_data_path, vectorizer_path):
    try:
        # Load model directly from pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Load the holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data['clean_comment'].fillna("")
        y_holdout = holdout_data['category']

        # Apply TF-IDF transformation
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict using the model
        y_pred = model.predict(X_holdout_tfidf_df)

        # Calculate performance metrics
        accuracy  = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average='weighted', zero_division=1)
        recall    = recall_score(y_holdout, y_pred, average='weighted', zero_division=1)
        f1        = f1_score(y_holdout, y_pred, average='weighted', zero_division=1)

        print(f"\n📊 Accuracy:  {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall:    {recall:.4f}")
        print(f"📊 F1 Score:  {f1:.4f}")

        # Assert thresholds
        assert accuracy  >= 0.40, f'Accuracy should be at least 0.40, got {accuracy}'
        assert precision >= 0.40, f'Precision should be at least 0.40, got {precision}'
        assert recall    >= 0.40, f'Recall should be at least 0.40, got {recall}'
        assert f1        >= 0.40, f'F1 score should be at least 0.40, got {f1}'

        print("✅ Performance test passed!")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")