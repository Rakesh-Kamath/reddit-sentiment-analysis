import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger('data_ingestion')
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


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        abs_path = os.path.join(PROJECT_ROOT, *data_path.split('/'))
        os.makedirs(abs_path, exist_ok=True)
        train_data.to_csv(os.path.join(abs_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(abs_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', abs_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        full_path = os.path.join(PROJECT_ROOT, params_path)
        with open(full_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug('Parameters retrieved from %s', full_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', full_path)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        logger.debug('Preprocessing complete')
        return df
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def main():
    try:
        params = load_params("params.yaml")
        test_size = params['data_ingestion']['test_size']

        df = load_data(
            "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )

        save_data(train_data, test_data, data_path="artifacts/data")

    except Exception as e:
        logger.error('Failed to complete data ingestion: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()