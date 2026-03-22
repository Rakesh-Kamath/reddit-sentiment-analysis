import os
import re
import logging
import joblib
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('sarcasm_detector')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SARCASM_MODELS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'sarcasm')

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    try:
        text = str(text).lower().strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(w for w in text.split() if w not in STOP_WORDS)
        text = ' '.join(LEMMATIZER.lemmatize(w) for w in text.split())
        return text
    except Exception as e:
        logger.error('Preprocessing error: %s', e)
        return str(text)


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
def load_sarcasm_data():
    logger.debug('Loading Reddit sarcasm dataset from HuggingFace...')
    ds = load_dataset('marcbishara/sarcasm-on-reddit')

    train_df = pd.DataFrame({
        'text':  ds['sft_train']['comment'],
        'label': ds['sft_train']['label']
    })
    test_df = pd.DataFrame({
        'text':  ds['sft_validation']['comment'],
        'label': ds['sft_validation']['label']
    })

    # Drop nulls and empty strings
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    train_df = train_df[train_df['text'].str.strip() != '']
    test_df  = test_df[test_df['text'].str.strip() != '']

    # Check label values
    logger.debug('Label distribution (train):\n%s', train_df['label'].value_counts())
    logger.debug('Train size: %d, Test size: %d', len(train_df), len(test_df))

    return train_df, test_df


# ---------------------------------------------------------------------------
# Train sarcasm detector
# ---------------------------------------------------------------------------
def train_sarcasm_detector(train_df: pd.DataFrame, test_df: pd.DataFrame):
    logger.debug('Preprocessing text...')
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)

    logger.debug('Fitting TF-IDF vectorizer...')
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test  = vectorizer.transform(test_df['clean_text'])

    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    logger.debug('Training Logistic Regression sarcasm classifier...')
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])

    logger.info('Sarcasm Detector Accuracy: %.4f', accuracy)
    logger.info('\n%s', report)

    return model, vectorizer, accuracy


# ---------------------------------------------------------------------------
# Save sarcasm model and vectorizer
# ---------------------------------------------------------------------------
def save_sarcasm_model(model, vectorizer):
    os.makedirs(SARCASM_MODELS_DIR, exist_ok=True)

    model_path      = os.path.join(SARCASM_MODELS_DIR, 'sarcasm_model.pkl')
    vectorizer_path = os.path.join(SARCASM_MODELS_DIR, 'sarcasm_vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    joblib.dump(vectorizer, vectorizer_path)

    logger.debug('Sarcasm model saved to %s', model_path)
    logger.debug('Sarcasm vectorizer saved to %s', vectorizer_path)


# ---------------------------------------------------------------------------
# Load sarcasm model
# ---------------------------------------------------------------------------
def load_sarcasm_model():
    model_path      = os.path.join(SARCASM_MODELS_DIR, 'sarcasm_model.pkl')
    vectorizer_path = os.path.join(SARCASM_MODELS_DIR, 'sarcasm_vectorizer.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


# ---------------------------------------------------------------------------
# Predict sarcasm probability for a list of texts
# ---------------------------------------------------------------------------
def predict_sarcasm_probability(texts: list, model=None, vectorizer=None) -> np.ndarray:
    if model is None or vectorizer is None:
        model, vectorizer = load_sarcasm_model()

    cleaned = [preprocess_text(t) for t in texts]
    X       = vectorizer.transform(cleaned)
    probs   = model.predict_proba(X)[:, 1]  # probability of being sarcastic
    return probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info('Starting sarcasm detector training...')

    train_df, test_df = load_sarcasm_data()
    model, vectorizer, accuracy = train_sarcasm_detector(train_df, test_df)
    save_sarcasm_model(model, vectorizer)

    logger.info('✅ Sarcasm detector trained and saved!')
    logger.info('📊 Accuracy: %.4f', accuracy)

    # Quick test
    test_texts = [
        "Oh great, another Monday morning",
        "The weather is nice today",
        "Yeah right, like that will ever happen",
        "I really love being stuck in traffic"
    ]
    probs = predict_sarcasm_probability(test_texts, model, vectorizer)
    for text, prob in zip(test_texts, probs):
        logger.info('Text: "%s" → Sarcasm probability: %.4f', text, prob)


if __name__ == '__main__':
    main()