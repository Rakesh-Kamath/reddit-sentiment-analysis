import os
import pytest
import pandas as pd
import joblib
import pickle
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.parametrize("model_path, vectorizer_path", [
    ("artifacts/models/lgbm_model.pkl", "artifacts/models/tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_path, vectorizer_path):
    try:
        # Load model directly from pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load the vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Create a dummy input
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict
        prediction = model.predict(input_df)

        # Verify input shape matches vectorizer features
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), \
            "Input feature count mismatch"

        # Verify output shape
        assert len(prediction) == input_df.shape[0], \
            "Output row count mismatch"

        print(f"✅ Model processed input successfully. Prediction: {prediction}")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")