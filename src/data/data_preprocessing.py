import os
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ['punkt', 'wordnet', 'stopwords', 'omw-1.4']:
    nltk.download(resource, quiet=True)

logger = logging.getLogger('data_preprocessing')
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

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet', 'very', 'too', 'won'}
LEMMATIZER = WordNetLemmatizer()

CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "where's": "where is", "who's": "who is",
    "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
}


def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'<.*?>', '', comment)
        comment = re.sub(r'@\w+', '', comment)
        comment = re.sub(r'#(\w+)', r'\1', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()
        for contraction, expansion in CONTRACTIONS.items():
            comment = comment.replace(contraction, expansion)
        comment = ' '.join(w for w in comment.split() if w not in STOP_WORDS)
        comment = ' '.join(LEMMATIZER.lemmatize(w, pos='v') for w in comment.split())
        comment = ' '.join(w for w in comment.split() if len(w) > 1)
        return re.sub(r'\s+', ' ', comment).strip()
    except Exception as e:
        logger.error('Error preprocessing comment: %s', e)
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
    logger.debug('Text normalisation complete')
    return df


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        abs_path = os.path.join(PROJECT_ROOT, *data_path.split('/'))
        os.makedirs(abs_path, exist_ok=True)
        train_data.to_csv(os.path.join(abs_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(abs_path, "test_processed.csv"), index=False)
        logger.debug('Processed data saved to %s', abs_path)
    except Exception as e:
        logger.error('Unexpected error saving data: %s', e)
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")
        train_path = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'train.csv')
        test_path  = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'test.csv')

        train_data = pd.read_csv(train_path)
        test_data  = pd.read_csv(test_path)
        logger.debug('Data loaded successfully')

        train_processed = normalize_text(train_data)
        test_processed  = normalize_text(test_data)

        save_data(train_processed, test_processed, data_path='artifacts/interim')

    except FileNotFoundError as e:
        logger.error('Data files not found: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete preprocessing: %s', e)
        raise


if __name__ == '__main__':
    main()