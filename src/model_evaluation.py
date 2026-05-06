import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


from logger import init_logging


LOGGER = init_logging(logger_name=os.path.basename(__file__),
                      log_file_path=os.getcwd()+"/logs/log_file.log")


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        LOGGER.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        LOGGER.error('File not found: %s', file_path)
        raise
    except Exception as e:
        LOGGER.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        LOGGER.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        LOGGER.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        LOGGER.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        LOGGER.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        LOGGER.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        LOGGER.debug('Metrics saved to %s', file_path)
    except Exception as e:
        LOGGER.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        LOGGER.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()