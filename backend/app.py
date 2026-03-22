import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import joblib
import pickle
import re
import pandas as pd
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
import nltk

# Load environment variables first
load_dotenv()

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

try:
    wn.ensure_loaded()
    stopwords.ensure_loaded()

    STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize('test')
    print("✅ NLTK Resources loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL NLTK ERROR: {e}")

app = Flask(__name__)
CORS(app)


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join([word for word in comment.split() if word not in STOP_WORDS])
        comment = ' '.join([LEMMATIZER.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer():
    """Load model and vectorizer from local pickle files."""
    try:
        base_dir = os.path.dirname(__file__)

        # Find vectorizer
        vectorizer_candidates = [
            os.path.join(base_dir, '..', 'artifacts', 'models', 'tfidf_vectorizer.pkl'),
            os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'),
            os.path.join(base_dir, 'tfidf_vectorizer.pkl'),
        ]
        vectorizer = None
        for path in vectorizer_candidates:
            if os.path.exists(path):
                vectorizer = joblib.load(path)
                print(f"✅ Vectorizer loaded from {path}")
                break

        # Find model
        model_candidates = [
            os.path.join(base_dir, '..', 'artifacts', 'models', 'lgbm_model.pkl'),
            os.path.join(base_dir, 'models', 'lgbm_model.pkl'),
            os.path.join(base_dir, 'lgbm_model.pkl'),
        ]
        model = None
        for path in model_candidates:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ Model loaded from {path}")
                break

        if model is None:
            raise FileNotFoundError("Could not find lgbm_model.pkl")
        if vectorizer is None:
            raise FileNotFoundError("Could not find tfidf_vectorizer.pkl")

        return model, vectorizer

    except Exception as e:
        print(f"❌ Error loading model and vectorizer: {e}")
        raise e


try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    print(f"CRITICAL: Failed to load model: {e}")
    model, vectorizer = None, None


def transform_comments(comments):
    """Preprocess and vectorize a list of comments."""
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
    else:
        feature_names = vectorizer.get_feature_names()
    return pd.DataFrame(transformed.toarray(), columns=feature_names)


@app.route('/')
def home():
    return jsonify({
        "message": "Reddit Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Predict sentiment for comments",
            "/predict_with_timestamps": "POST - Predict with timestamps",
            "/generate_chart": "POST - Generate pie chart",
            "/generate_wordcloud": "POST - Generate word cloud",
            "/generate_trend_graph": "POST - Generate trend graph",
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503

    comments = request.json.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        df = transform_comments(comments)
        predictions = [str(p) for p in model.predict(df.values).tolist()]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503

    data = request.json.get('comments')
    if not data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in data]
        timestamps = [item['timestamp'] for item in data]
        df = transform_comments(comments)
        predictions = [str(p) for p in model.predict(df.values).tolist()]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        sentiment_counts = request.get_json().get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
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
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        comments = request.get_json().get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)

        wc = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


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
        totals = monthly.sum(axis=1)
        pct = (monthly.T / totals).T * 100

        for v in [-1, 0, 1]:
            if v not in pct.columns:
                pct[v] = 0
        pct = pct[[-1, 0, 1]]

        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        plt.figure(figsize=(12, 6))
        for v in [-1, 0, 1]:
            plt.plot(pct.index, pct[v], marker='o', label=labels[v], color=colors[v])

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
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
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)