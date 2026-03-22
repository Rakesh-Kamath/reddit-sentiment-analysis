import matplotlib
matplotlib.use('Agg')

import os
import re
import sys
import io
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mlflow
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

try:
    wn.ensure_loaded()
    stopwords.ensure_loaded()
    STOP_WORDS  = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    LEMMATIZER  = WordNetLemmatizer()
    LEMMATIZER.lemmatize('test')
    print("✅ NLTK resources loaded.")
except Exception as e:
    print(f"❌ NLTK error: {e}")

# Read credentials from environment (not hardcoded)
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', '').strip()
DAGSHUB_TOKEN    = os.getenv('DAGSHUB_TOKEN', '').strip()
REPO_NAME        = os.getenv('REPO_NAME', 'reddit-sentiment-analysis').strip()

if DAGSHUB_TOKEN:
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
else:
    print("WARNING: DAGSHUB_TOKEN not set – MLflow tracking may fail.")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join(w for w in comment.split() if w not in STOP_WORDS)
        comment = ' '.join(LEMMATIZER.lemmatize(w) for w in comment.split())
        return comment
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return comment

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_version}"
    model     = mlflow.pyfunc.load_model(model_uri)

    # Try several likely locations for the vectorizer
    base_dir = os.path.dirname(__file__)
    candidates = [
        vectorizer_path,
        os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'),
        os.path.join(base_dir, '..', 'artifacts', 'models', 'tfidf_vectorizer.pkl'),
    ]
    for path in candidates:
        if os.path.exists(path):
            vectorizer = joblib.load(path)
            print(f"✅ Vectorizer loaded from {path}")
            return model, vectorizer

    raise FileNotFoundError(f"Vectorizer not found. Tried: {candidates}")


try:
    model, vectorizer = load_model_and_vectorizer(
        "reddit_sentiment_lgbm",
        "Staging",
        "models/tfidf_vectorizer.pkl",
    )
except Exception as e:
    print(f"WARNING: Model load failed: {e}")
    model, vectorizer = None, None

# ---------------------------------------------------------------------------
# Helper: transform text → DataFrame for prediction
# ---------------------------------------------------------------------------
def transform_comments(comments):
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed  = vectorizer.transform(preprocessed)
    feature_names = (
        vectorizer.get_feature_names_out()
        if hasattr(vectorizer, 'get_feature_names_out')
        else vectorizer.get_feature_names()
    )
    return pd.DataFrame(transformed.toarray(), columns=feature_names)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({
        "message": "Reddit Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 503

    comments = request.json.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        df          = transform_comments(comments)
        predictions = [str(p) for p in model.predict(df).tolist()]
        return jsonify([{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json.get('comments')
    if not data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments   = [item['text'] for item in data]
        timestamps = [item['timestamp'] for item in data]
        df         = transform_comments(comments)
        predictions = [str(p) for p in model.predict(df).tolist()]
        return jsonify([
            {"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(comments, predictions, timestamps)
        ])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        sentiment_counts = request.get_json().get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes  = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "All sentiment counts are zero"}), 400

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {e}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        comments = request.get_json().get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        text = ' '.join(preprocess_comment(c) for c in comments)
        wc   = WordCloud(width=800, height=400, background_color='black',
                         colormap='Blues', collocations=False).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Word cloud failed: {e}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        sentiment_data = request.get_json().get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment'] = df['sentiment'].astype(int)
        df.set_index('timestamp', inplace=True)

        monthly = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        totals  = monthly.sum(axis=1)
        pct     = (monthly.T / totals).T * 100

        for v in [-1, 0, 1]:
            if v not in pct.columns:
                pct[v] = 0
        pct = pct[[-1, 0, 1]]

        colors  = {-1: 'red', 0: 'gray', 1: 'green'}
        labels  = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        plt.figure(figsize=(12, 6))
        for v in [-1, 0, 1]:
            plt.plot(pct.index, pct[v], marker='o', label=labels[v], color=colors[v])

        plt.title('Monthly Sentiment % Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Trend graph failed: {e}"}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=True)