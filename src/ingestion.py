"""data ingestion module"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import init_logging
from read_params import load_params

PARAMS = load_params(os.path.join(os.getcwd()+"/params.yaml"))


LOGGER = init_logging(logger_name=os.path.basename(__file__),
                      log_file_path=os.getcwd()+"/logs/log_file.log")


def load_data(path: str) -> pd.DataFrame:
    """load data from path"""

    try:
        data = pd.read_csv(path)
        return data
        LOGGER.debug('Data loaded from %s ', path)
    except pd.errors.ParserError as e:
        LOGGER.error('Failed to parse CSV file: %s', e)
    except Exception as e:
        LOGGER.error('Some error occured while loading the data: %s', e)



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        LOGGER.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        LOGGER.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        LOGGER.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        LOGGER.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        LOGGER.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        test_size = PARAMS['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(path=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        LOGGER.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
