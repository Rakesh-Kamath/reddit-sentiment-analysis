import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
import sys
import nltk
from mlflow.tracking import MlflowClient

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

DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', '').strip()
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '').strip()
REPO_NAME = os.getenv('REPO_NAME', 'reddit-sentiment-analysis').strip()

if DAGSHUB_TOKEN:
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
else:
    print("WARNING: DAGSHUB_TOKEN not found in environment variables")

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


def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load model and vectorizer from MLflow and local storage."""
    try:
        tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'),
            os.path.join(base_dir, '..', 'artifacts', 'models', 'tfidf_vectorizer.pkl'),
            vectorizer_path,
        ]
        for path in candidates:
            if os.path.exists(path):
                vectorizer = joblib.load(path)
                print(f"✅ Model and vectorizer loaded successfully from {path}")
                return model, vectorizer

        raise FileNotFoundError(f"Vectorizer not found. Tried: {candidates}")

    except Exception as e:
        print(f"❌ Error loading model and vectorizer: {e}")
        raise e


try:
    model, vectorizer = load_model_and_vectorizer(
        "reddit_sentiment_lgbm",
        "Staging",
        "models/tfidf_vectorizer.pkl"
    )
except Exception as e:
    print(f"CRITICAL: Failed to load model: {e}")
    model, vectorizer = None, None


@app.route('/')
def home():
    return jsonify({
        "message": "Reddit Sentiment Analysis API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/": "This message",
            "/predict": "POST - Predict sentiment for comments"
        }
    })


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)

        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()

        transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        predictions = model.predict(transformed_comments_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)

        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()

        transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        predictions = model.predict(transformed_comments_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": category} for comment, category in zip(comments, predictions)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

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
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)