# Feature Engineering Pipeline for Text Classification
# This script performs TF-IDF vectorization on text data for machine learning models
# Author: Rajat Bansal
# Date: September 29, 2025

# Import necessary libraries for data processing and machine learning
import pandas as pd              # For data manipulation and analysis
import os                       # For operating system interface and file path operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical features
import logging                  # For comprehensive logging and debugging
import yaml                    # For reading configuration parameters from YAML files


# ===== DIRECTORY SETUP =====
# Ensure the "logs" directory exists for storing log files
# This prevents FileNotFoundError when creating log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # exist_ok=True prevents error if directory already exists


# ===== LOGGING CONFIGURATION =====
# Set up comprehensive logging system for debugging and monitoring
# This helps track the execution flow and identify issues during processing

# Create a logger instance with a specific name for this module
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')  # Set logging level to DEBUG to capture all log messages


# Configure console handler to display logs in the terminal/console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Console will show DEBUG level and above messages


# Configure file handler to save logs to a permanent file
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # File will store DEBUG level and above messages


# Set up log message formatting to include timestamp, logger name, level, and message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)  # Apply formatter to console output
file_handler.setFormatter(formatter)     # Apply formatter to file output


# Add both handlers to the logger so messages go to both console and file
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ===== PARAMETER LOADING FUNCTION =====
def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.
    
    This function reads hyperparameters and configuration settings from a YAML file,
    which allows for easy parameter tuning without modifying the code.
    
    Args:
        params_path (str): Path to the YAML parameter file
        
    Returns:
        dict: Dictionary containing all parameters from the YAML file
        
    Raises:
        FileNotFoundError: If the parameter file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        Exception: For any other unexpected errors
    """
    try:
        # Open and read the YAML file safely
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)  # safe_load prevents execution of arbitrary code
        
        # Log successful parameter loading for debugging
        logger.debug('Parameters retrieved from %s', params_path)
        return params
        
    except FileNotFoundError:
        # Handle case where parameter file doesn't exist
        logger.error('File not found: %s', params_path)
        raise  # Re-raise the exception to stop execution
        
    except yaml.YAMLError as e:
        # Handle YAML parsing errors (malformed YAML syntax)
        logger.error('YAML error: %s', e)
        raise
        
    except Exception as e:
        # Catch any other unexpected errors during parameter loading
        logger.error('Unexpected error: %s', e)
        raise


# ===== DATA LOADING FUNCTION =====
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file.
    
    This function reads CSV data and handles missing values by filling them with empty strings.
    This preprocessing step is crucial for text processing pipelines.
    
    Args:
        file_path (str): Path to the CSV file containing the data
        
    Returns:
        pd.DataFrame: Loaded DataFrame with NaN values filled with empty strings
        
    Raises:
        pd.errors.ParserError: If the CSV file is malformed or corrupted
        Exception: For any other file reading errors
    """
    try:
        # Load CSV file into pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Fill NaN/missing values with empty strings to prevent errors in text processing
        # This is important because TfidfVectorizer cannot handle NaN values
        df.fillna('', inplace=True)  # inplace=True modifies the DataFrame directly
        
        # Log successful data loading for debugging
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
        
    except pd.errors.ParserError as e:
        # Handle CSV parsing errors (corrupted or malformed CSV files)
        logger.error('Failed to parse the CSV file: %s', e)
        raise
        
    except Exception as e:
        # Handle any other file reading errors (permission issues, disk errors, etc.)
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# ===== TF-IDF TRANSFORMATION FUNCTION =====
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to text data.
    
    TF-IDF converts text documents into numerical feature vectors that machine learning
    algorithms can process. It measures word importance by considering both term frequency
    in documents and rarity across the entire corpus.
    
    Args:
        train_data (pd.DataFrame): Training dataset containing 'text' and 'target' columns
        test_data (pd.DataFrame): Test dataset containing 'text' and 'target' columns  
        max_features (int): Maximum number of features (words) to include in the vocabulary
        
    Returns:
        tuple: (train_df, test_df) - DataFrames with TF-IDF features and labels
        
    Raises:
        Exception: For any errors during the vectorization process
    """
    try:
        # Initialize TF-IDF vectorizer with specified maximum features
        # max_features limits vocabulary size to prevent overfitting and reduce memory usage
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract text and target columns from training data
        # .values converts pandas Series to numpy arrays for better performance
        X_train = train_data['text'].values    # Training text data
        y_train = train_data['target'].values  # Training labels/targets
        X_test = test_data['text'].values      # Test text data  
        y_test = test_data['target'].values    # Test labels/targets

        # Fit the vectorizer on training data and transform both train and test sets
        # fit_transform(): learns vocabulary from training data AND transforms it
        X_train_bow = vectorizer.fit_transform(X_train)
        
        # transform(): applies the learned vocabulary to test data (no learning here)
        # This ensures consistent feature space between train and test sets
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse matrices to dense arrays and create DataFrames
        # toarray() converts scipy sparse matrix to numpy dense array
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train  # Add target labels as 'label' column

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test    # Add target labels as 'label' column

        # Log successful transformation for debugging
        logger.debug('tfidf applied and data transformed')
        return train_df, test_df
        
    except Exception as e:
        # Handle any errors during TF-IDF transformation
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


# ===== DATA SAVING FUNCTION =====
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file with proper directory creation.
    
    This function ensures the output directory exists before saving the file,
    preventing FileNotFoundError during the save operation.
    
    Args:
        df (pd.DataFrame): DataFrame to be saved
        file_path (str): Complete path where the CSV file should be saved
        
    Returns:
        None
        
    Raises:
        Exception: For any file writing errors (disk space, permissions, etc.)
    """
    try:
        # Create the directory structure if it doesn't exist
        # os.path.dirname() extracts the directory path from the full file path
        # exist_ok=True prevents error if directory already exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save DataFrame to CSV file without including the index column
        # index=False prevents pandas from adding an unnecessary index column
        df.to_csv(file_path, index=False)
        
        # Log successful data saving for debugging
        logger.debug('Data saved to %s', file_path)
        
    except Exception as e:
        # Handle any file saving errors (disk space, permissions, path issues)
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# ===== MAIN EXECUTION FUNCTION =====
def main():
    """
    Main execution function that orchestrates the entire feature engineering pipeline.
    
    This function coordinates all the steps:
    1. Load parameters from configuration file
    2. Load training and test data
    3. Apply TF-IDF transformation
    4. Save processed data for model training
    
    The entire process is wrapped in try-catch for robust error handling.
    """
    try:
        # Step 1: Load configuration parameters from YAML file
        # Parameters file contains hyperparameters and settings for the pipeline
        params = load_params(params_path='params.yaml')
        
        # Extract the max_features parameter for TF-IDF vectorization
        # This controls the vocabulary size and model complexity
        max_features = params['feature_engineering']['max_features']
        # Alternative: max_features = 50  # Hardcoded value for testing

        # Step 2: Load preprocessed training and test datasets
        # These files should contain 'text' and 'target' columns from previous preprocessing steps
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Step 3: Apply TF-IDF transformation to convert text to numerical features
        # This creates feature vectors that machine learning algorithms can process
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Step 4: Save the transformed datasets for model training phase
        # These processed files will be used by the model training pipeline
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
        
    except Exception as e:
        # Handle any errors that occur during the entire pipeline execution
        # Log the error and also print to console for immediate visibility
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


# ===== SCRIPT ENTRY POINT =====
# This ensures the main() function only runs when the script is executed directly,
# not when it's imported as a module in other scripts
if __name__ == '__main__':
    main()



# import pandas as pd
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# import logging
# import yaml

# # Ensure the "logs" directory exists
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# # logging configuration
# logger = logging.getLogger('feature_engineering')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir, 'feature_engineering.log')
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel('DEBUG')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logger.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logger.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logger.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error: %s', e)
#         raise

# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         df.fillna('', inplace=True)
#         logger.debug('Data loaded and NaNs filled from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
#     """Apply TfIdf to the data."""
#     try:
#         vectorizer = TfidfVectorizer(max_features=max_features)

#         X_train = train_data['text'].values
#         y_train = train_data['target'].values
#         X_test = test_data['text'].values
#         y_test = test_data['target'].values

#         X_train_bow = vectorizer.fit_transform(X_train)
#         X_test_bow = vectorizer.transform(X_test)

#         train_df = pd.DataFrame(X_train_bow.toarray())
#         train_df['label'] = y_train

#         test_df = pd.DataFrame(X_test_bow.toarray())
#         test_df['label'] = y_test

#         logger.debug('tfidf applied and data transformed')
#         return train_df, test_df
#     except Exception as e:
#         logger.error('Error during Bag of Words transformation: %s', e)
#         raise

# def save_data(df: pd.DataFrame, file_path: str) -> None:
#     """Save the dataframe to a CSV file."""
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         df.to_csv(file_path, index=False)
#         logger.debug('Data saved to %s', file_path)
#     except Exception as e:
#         logger.error('Unexpected error occurred while saving the data: %s', e)
#         raise

# def main():
#     try:
#         params = load_params(params_path='./../params.yaml')
#         max_features = params['feature_engineering']['max_features']
#         # max_features = 50

#         train_data = load_data('./data/interim/train_processed.csv')
#         test_data = load_data('./data/interim/test_processed.csv')

#         train_df, test_df = apply_tfidf(train_data, test_data, max_features)

#         save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
#         save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
#     except Exception as e:
#         logger.error('Failed to complete the feature engineering process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
