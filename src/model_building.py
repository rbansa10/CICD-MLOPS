
"""
Machine Learning Model Building Pipeline for Text Classification
===============================================================
This script implements a complete machine learning model building pipeline using
Random Forest classifier for text classification tasks. It handles:
- Configuration parameter loading from YAML files
- Preprocessed TF-IDF feature data loading
- Random Forest model training with hyperparameters
- Model serialization and persistence using pickle
- Comprehensive logging and error handling

Author: Machine Learning Engineering Team
Date: September 29, 2025
Version: 1.0 (Production-ready model building pipeline)
"""

# ===== IMPORT STATEMENTS =====
# Standard library imports for system operations and file handling
import os                                    # Operating system interface for file/directory operations
import numpy as np                          # Numerical computing library for array operations
import pandas as pd                         # Data manipulation and analysis library
import pickle                              # Python object serialization for model persistence
import logging                             # Comprehensive logging system for monitoring

# Machine learning library imports
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm implementation

# Configuration file handling
import yaml                                # YAML file parser for configuration parameters


# ===== LOGGING DIRECTORY SETUP =====
# Create directory structure for storing comprehensive log files
# This ensures all model building activities are tracked and auditable
log_dir = 'logs'                           # Standard directory name for all log files
os.makedirs(log_dir, exist_ok=True)       # Create logs directory if it doesn't exist
                                          # exist_ok=True prevents errors if directory already exists


# ===== COMPREHENSIVE LOGGING CONFIGURATION =====
# Set up sophisticated logging system for model building monitoring
# This provides both real-time feedback and permanent audit trails

# Create named logger instance specifically for model building operations
logger = logging.getLogger('model_building')  # Module-specific logger identifier
logger.setLevel('DEBUG')                       # Capture all log levels from DEBUG upward
                                              # DEBUG < INFO < WARNING < ERROR < CRITICAL


# Configure console handler for real-time terminal output during training
console_handler = logging.StreamHandler()     # Handler for stdout/terminal output
console_handler.setLevel('DEBUG')             # Display DEBUG level and above in console
                                              # Useful for monitoring training progress


# Configure file handler for persistent log storage and audit trails
log_file_path = os.path.join(log_dir, 'model_building.log')  # Complete path to log file
file_handler = logging.FileHandler(log_file_path)           # Handler for file-based logging
file_handler.setLevel('DEBUG')                              # Store DEBUG level and above in file
                                                            # Enables post-training analysis


# Create standardized log message format with comprehensive context
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Format components: timestamp - logger_name - severity_level - actual_message
# This format enables easy parsing and analysis of log files

# Apply consistent formatting to both output destinations
console_handler.setFormatter(formatter)       # Format console messages for readability
file_handler.setFormatter(formatter)         # Format file messages for parsing


# Register both handlers with the logger for dual output capability
logger.addHandler(console_handler)           # Enable real-time console monitoring
logger.addHandler(file_handler)             # Enable persistent file logging


# ===== CONFIGURATION PARAMETER LOADING FUNCTION =====
def load_params(params_path: str) -> dict:
    """
    Load machine learning hyperparameters and configuration from YAML file.
    
    This function provides centralized configuration management for the ML pipeline,
    enabling easy hyperparameter tuning without code modifications. YAML format
    allows for hierarchical parameter organization and human-readable configuration.
    
    Args:
        params_path (str): File system path to the YAML configuration file
                          Example: 'params.yaml' or 'config/model_params.yaml'
        
    Returns:
        dict: Nested dictionary containing all configuration parameters
              Structure typically includes model hyperparameters, file paths,
              and training settings
              
    Raises:
        FileNotFoundError: When the specified YAML file doesn't exist
        yaml.YAMLError: When the YAML file contains syntax errors or is malformed
        Exception: For any other unexpected errors during file processing
        
    Example Configuration File Structure:
        model_building:
          n_estimators: 100
          random_state: 42
          max_depth: 10
    """
    try:
        # Open and parse YAML file using safe loading to prevent code execution
        with open(params_path, 'r') as file:
            # safe_load prevents execution of arbitrary Python code embedded in YAML
            # This is a security best practice for configuration file parsing
            params = yaml.safe_load(file)
            
        # Log successful parameter loading with file path for debugging
        logger.debug('Parameters retrieved from %s', params_path)
        return params
        
    except FileNotFoundError:
        # Handle missing configuration file with specific error logging
        logger.error('File not found: %s', params_path)
        raise  # Re-raise exception to halt execution with proper error context
        
    except yaml.YAMLError as e:
        # Handle YAML syntax errors or malformed file structure
        logger.error('YAML error: %s', e)
        raise  # Re-raise to provide specific YAML error information
        
    except Exception as e:
        # Catch-all for unexpected errors during configuration loading
        logger.error('Unexpected error: %s', e)
        raise  # Re-raise to maintain error context and halt execution


# ===== DATA LOADING FUNCTION =====
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed machine learning data from CSV file with validation.
    
    This function loads TF-IDF processed feature data that was created by the
    feature engineering pipeline. It includes data validation and shape logging
    for debugging and monitoring purposes.
    
    Args:
        file_path (str): Complete path to the CSV file containing processed features
                        Expected format: rows=samples, columns=features+label
                        Example: './data/processed/train_tfidf.csv'
    
    Returns:
        pd.DataFrame: Loaded dataset with features and labels
                     Shape: (n_samples, n_features + 1)
                     Last column typically contains target labels
    
    Raises:
        pd.errors.ParserError: When CSV file is corrupted or has formatting issues
        FileNotFoundError: When the specified file path doesn't exist
        Exception: For any other file reading or parsing errors
        
    Data Validation:
        - Logs dataset shape for verification
        - Ensures successful parsing of CSV format
        - Provides detailed error context for troubleshooting
    """
    try:
        # Load CSV file into pandas DataFrame with default parsing settings
        df = pd.read_csv(file_path)
        
        # Log successful loading with dataset dimensions for validation
        # Shape information helps verify data integrity and expected format
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
        
    except pd.errors.ParserError as e:
        # Handle CSV parsing errors (corrupted files, encoding issues)
        logger.error('Failed to parse the CSV file: %s', e)
        raise  # Re-raise with parsing context for debugging
        
    except FileNotFoundError as e:
        # Handle missing data file with specific path information
        logger.error('File not found: %s', e)
        raise  # Re-raise to halt execution with clear file path context
        
    except Exception as e:
        # Handle any other unexpected errors during data loading
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise  # Re-raise to preserve error information and context


# ===== MODEL TRAINING FUNCTION =====
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train Random Forest classifier with specified hyperparameters.
    
    This function implements the core machine learning model training using
    Random Forest algorithm. Random Forest is chosen for its robustness,
    interpretability, and excellent performance on text classification tasks
    with TF-IDF features.
    
    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features)
                             Contains TF-IDF vectors representing text documents
        y_train (np.ndarray): Training target vector of shape (n_samples,)
                             Contains encoded class labels (0, 1, 2, etc.)
        params (dict): Hyperparameter dictionary containing model configuration
                      Expected keys: 'n_estimators', 'random_state', etc.
    
    Returns:
        RandomForestClassifier: Fully trained Random Forest model ready for prediction
                               Model includes learned decision trees and feature importances
    
    Raises:
        ValueError: When feature and label arrays have mismatched sample counts
        Exception: For any other errors during model training process
        
    Training Details:
        - Validates input data shape consistency
        - Uses specified hyperparameters from configuration
        - Logs training progress for monitoring
        - Returns trained model ready for evaluation/prediction
    """
    try:
        # Validate that feature matrix and label vector have matching sample counts
        # This is crucial for supervised learning - each sample needs a corresponding label
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        # Log model initialization with hyperparameters for debugging and reproducibility
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        
        # Initialize Random Forest classifier with specified hyperparameters
        # n_estimators: number of decision trees in the forest (more trees = better performance but slower)
        # random_state: ensures reproducible results across different runs
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],    # Number of trees in the forest
            random_state=params['random_state']     # Seed for reproducible random operations
        )
        
        # Log training initiation with sample count for monitoring progress
        logger.debug('Model training started with %d samples', X_train.shape[0])
        
        # Execute model training using scikit-learn's fit method
        # This builds all decision trees and learns feature importance rankings
        clf.fit(X_train, y_train)
        
        # Log successful training completion for monitoring and debugging
        logger.debug('Model training completed')
        
        return clf  # Return fully trained model ready for evaluation
        
    except ValueError as e:
        # Handle data validation errors with specific error context
        logger.error('ValueError during model training: %s', e)
        raise  # Re-raise to halt execution with validation error details
        
    except Exception as e:
        # Handle any other training errors (memory issues, algorithm failures)
        logger.error('Error during model training: %s', e)
        raise  # Re-raise to preserve error context for debugging


# ===== MODEL PERSISTENCE FUNCTION =====
def save_model(model, file_path: str) -> None:
    """
    Serialize and save trained model to disk using pickle format.
    
    This function handles model persistence, allowing trained models to be
    stored permanently and loaded later for prediction or evaluation.
    Pickle format preserves the complete model state including learned parameters.
    
    Args:
        model: Trained machine learning model object (typically RandomForestClassifier)
               Must be a serializable scikit-learn model with learned parameters
        file_path (str): Complete file system path for saving the model
                        Example: 'models/random_forest_classifier.pkl'
                        
    Returns:
        None: Function performs file I/O operation without return value
        
    Raises:
        FileNotFoundError: When the specified directory path doesn't exist
        Exception: For any other file writing or serialization errors
        
    File Operations:
        - Creates directory structure if it doesn't exist
        - Uses binary mode writing for pickle serialization
        - Logs successful save operation with file path
    """
    try:
        # Ensure the target directory exists before attempting to save
        # os.path.dirname extracts directory path from full file path
        # makedirs with exist_ok=True creates directory structure if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Open file in binary write mode for pickle serialization
        # Binary mode is required for pickle format which stores binary data
        with open(file_path, 'wb') as file:
            # Serialize the trained model object to binary format
            # pickle.dump preserves all model state including learned parameters
            pickle.dump(model, file)
            
        # Log successful model saving with file path for confirmation
        logger.debug('Model saved to %s', file_path)
        
    except FileNotFoundError as e:
        # Handle directory path errors with specific context
        logger.error('File path not found: %s', e)
        raise  # Re-raise to halt execution with path error details
        
    except Exception as e:
        # Handle any other file writing or serialization errors
        logger.error('Error occurred while saving the model: %s', e)
        raise  # Re-raise to preserve error context for debugging


# ===== MAIN EXECUTION ORCHESTRATION FUNCTION =====
def main():
    """
    Main orchestration function for the complete model building pipeline.
    
    This function coordinates all phases of model building from configuration
    loading through model training and persistence. It implements the complete
    ML workflow with comprehensive error handling and logging.
    
    Pipeline Stages:
    1. Load hyperparameters from YAML configuration file
    2. Load preprocessed training data (TF-IDF features + labels)
    3. Prepare feature matrix and target vector for training
    4. Train Random Forest model with specified hyperparameters
    5. Save trained model to disk for later use
    
    File Structure Expectations:
        params.yaml                           # Configuration file with hyperparameters
        ./data/processed/train_tfidf.csv     # Preprocessed training features
        models/model.pkl                     # Output location for trained model
        
    Error Handling:
        - Comprehensive try-catch for entire pipeline
        - Specific error logging for each pipeline stage
        - User-friendly error messages for debugging
    """
    try:
        # ===== STAGE 1: CONFIGURATION LOADING =====
        # Load hyperparameters and configuration settings from YAML file
        # ['model_building'] extracts the specific section for this pipeline stage
        params = load_params('./../params.yaml')['model_building']
        
        # ===== STAGE 2: TRAINING DATA LOADING =====
        # Load preprocessed TF-IDF feature data from feature engineering pipeline
        # This file contains numerical features ready for machine learning
        train_data = load_data('./data/processed/train_tfidf.csv')
        
        # ===== STAGE 3: DATA PREPARATION =====
        # Prepare feature matrix (X) and target vector (y) for supervised learning
        # iloc[:, :-1] selects all columns except the last one (features)
        # iloc[:, -1] selects only the last column (target labels)
        X_train = train_data.iloc[:, :-1].values  # Feature matrix: (n_samples, n_features)
        y_train = train_data.iloc[:, -1].values   # Target vector: (n_samples,)

        # ===== STAGE 4: MODEL TRAINING =====
        # Train Random Forest classifier using prepared data and hyperparameters
        clf = train_model(X_train, y_train, params)
        
        # ===== STAGE 5: MODEL PERSISTENCE =====
        # Save trained model to standard location for later use in prediction pipeline
        model_save_path = 'models/model.pkl'     # Standard model file location
        save_model(clf, model_save_path)
        
        # Log successful completion of entire pipeline
        print("✅ Model building completed successfully!")
        print(f"📁 Model saved to: {model_save_path}")

    except Exception as e:
        # ===== COMPREHENSIVE ERROR HANDLING =====
        # Handle any errors that occur during the entire model building process
        # Provides both technical logging and user-friendly error messages
        logger.error('Failed to complete the model building process: %s', e)
        print(f"❌ Error: {e}")
        print("🔍 Check logs for detailed error information")


# ===== SCRIPT ENTRY POINT =====
# Ensures main() function only executes when script is run directly,
# not when imported as a module by other scripts
if __name__ == '__main__':
    # Execute the complete model building pipeline
    main()

# import os
# import numpy as np
# import pandas as pd
# import pickle
# import logging
# from sklearn.ensemble import RandomForestClassifier
# import yaml

# # Ensure the "logs" directory exists
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# # logging configuration
# logger = logging.getLogger('model_building')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir, 'model_building.log')
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
#     """
#     Load data from a CSV file.
    
#     :param file_path: Path to the CSV file
#     :return: Loaded DataFrame
#     """
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except FileNotFoundError as e:
#         logger.error('File not found: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
#     """
#     Train the RandomForest model.
    
#     :param X_train: Training features
#     :param y_train: Training labels
#     :param params: Dictionary of hyperparameters
#     :return: Trained RandomForestClassifier
#     """
#     try:
#         if X_train.shape[0] != y_train.shape[0]:
#             raise ValueError("The number of samples in X_train and y_train must be the same.")
        
#         logger.debug('Initializing RandomForest model with parameters: %s', params)
#         clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
#         logger.debug('Model training started with %d samples', X_train.shape[0])
#         clf.fit(X_train, y_train)
#         logger.debug('Model training completed')
        
#         return clf
#     except ValueError as e:
#         logger.error('ValueError during model training: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Error during model training: %s', e)
#         raise


# def save_model(model, file_path: str) -> None:
#     """
#     Save the trained model to a file.
    
#     :param model: Trained model object
#     :param file_path: Path to save the model file
#     """
#     try:
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
#         with open(file_path, 'wb') as file:
#             pickle.dump(model, file)
#         logger.debug('Model saved to %s', file_path)
#     except FileNotFoundError as e:
#         logger.error('File path not found: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Error occurred while saving the model: %s', e)
#         raise

# def main():
#     try:
#         params = load_params('params.yaml')['model_building']
#         train_data = load_data('./data/processed/train_tfidf.csv')
#         X_train = train_data.iloc[:, :-1].values
#         y_train = train_data.iloc[:, -1].values

#         clf = train_model(X_train, y_train, params)
        
#         model_save_path = 'models/model.pkl'
#         save_model(clf, model_save_path)

#     except Exception as e:
#         logger.error('Failed to complete the model building process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
