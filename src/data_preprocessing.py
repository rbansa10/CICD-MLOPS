import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
import tkinter as tk
from tkinter import filedialog

def pick_file(title="Select CSV file"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path
def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # GUI prompt to pick train and test files
        print("Pick the TRAIN CSV file:")
        train_file = pick_file("Select TRAIN CSV file")
        print(f"TRAIN file selected: {train_file}")  # Print train file path
        print("Pick the TEST CSV file:")
        #print (os.path(train_file))
        test_file = pick_file("Select TEST CSV file")
        print(f"TEST file selected: {test_file}")    # Print test file path
        if not train_file or not os.path.isfile(train_file):
            logger.error('Required file not found: %s', train_file)
            print(f"Error: Required file not found: {train_file}")
            return
        if not test_file or not os.path.isfile(test_file):
            logger.error('Required file not found: %s', test_file)
            print(f"Error: Required file not found: {test_file}")
            return

        # Fetch the data from selected files
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

# def main(text_column='text', target_column='target'):
#     """
#     Main function to load raw data, preprocess it, and save the processed data.
#     """
#     try:
#         # Fetch the data from data/raw
#         #CICD-MLOPS/src/data_preprocessing.py
#         train_data = pd.read_csv('./data/raw/train.csv')
#         test_data = pd.read_csv('./data/raw/test.csv')
#         logger.debug('Data loaded properly')

#         # Transform the data
#         train_processed_data = preprocess_df(train_data, text_column, target_column)
#         test_processed_data = preprocess_df(test_data, text_column, target_column)

#         # Store the data inside data/processed
#         data_path = os.path.join("./data", "interim")
#         os.makedirs(data_path, exist_ok=True)
        
#         train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
#         test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
#         logger.debug('Processed data saved to %s', data_path)
#     except FileNotFoundError as e:
#         logger.error('File not found: %s', e)
#     except pd.errors.EmptyDataError as e:
#         logger.error('No data: %s', e)
#     except Exception as e:
#         logger.error('Failed to complete the data transformation process: %s', e)
#         print(f"Error: {e}")

if __name__ == '__main__':
    main()
