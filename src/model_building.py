import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

from logger import init_logging
LOGGER = init_logging(logger_name=os.path.basename(__file__),
                      log_file_path=os.getcwd()+"/logs/log_file.log")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        LOGGER.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        LOGGER.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        LOGGER.error('File not found: %s', e)
        raise
    except Exception as e:
        LOGGER.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        params = {'n_estimators': 22, 'random_state': 2}
        LOGGER.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        LOGGER.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        LOGGER.debug('Model training completed')
        
        return clf
    except ValueError as e:
        LOGGER.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        LOGGER.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        LOGGER.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        LOGGER.error('File path not found: %s', e)
        raise
    except Exception as e:
        LOGGER.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        clf = train_model(X_train, y_train)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        LOGGER.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
