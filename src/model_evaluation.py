"""
Machine Learning Model Evaluation Pipeline with MLOps Integration
================================================================
This script implements a comprehensive model evaluation and experiment tracking system
for machine learning pipelines. It provides:
- Trained model loading and validation
- Multi-metric evaluation on test datasets
- Experiment tracking with DVC Live integration
- Performance metrics persistence and reporting
- Comprehensive logging and error handling for production deployment

Author: Rajat Bansal
Date: September 29, 2025
Version: 2.0 (Production-ready with experiment tracking)
Dependencies: scikit-learn, pandas, numpy, dvc, dvclive
"""

# ===== IMPORT STATEMENTS =====
# Standard library imports for system operations and data handling
import os                                    # Operating system interface for file/directory operations
import numpy as np                          # Numerical computing library for array operations and metrics
import pandas as pd                         # Data manipulation library for CSV handling
import pickle                              # Object serialization for model loading
import json                                # JSON format handling for metrics persistence

# Machine learning evaluation metrics from scikit-learn
from sklearn.metrics import (
    accuracy_score,      # Overall classification accuracy (correct predictions / total predictions)
    precision_score,     # Positive predictive value (true positives / (true positives + false positives))
    recall_score,        # Sensitivity or true positive rate (true positives / (true positives + false negatives))
    roc_auc_score       # Area Under ROC Curve for binary classification performance
)

# Logging and configuration management
import logging                             # Comprehensive logging system for monitoring
import yaml                               # YAML parser for configuration file handling

# MLOps and experiment tracking integration
from dvclive import Live                  # DVC Live for experiment tracking and versioning
                                         # Enables reproducible ML experiments and model comparison


# ===== LOGGING DIRECTORY SETUP =====
# Create standardized directory structure for comprehensive audit trails
# This ensures all evaluation activities are properly logged and traceable
log_dir = 'logs'                          # Standard directory for all application logs
os.makedirs(log_dir, exist_ok=True)      # Create logs directory with error prevention
                                         # exist_ok=True avoids errors if directory already exists


# ===== ADVANCED LOGGING CONFIGURATION =====
# Configure sophisticated dual-output logging system for model evaluation
# This provides both real-time monitoring and persistent audit capabilities

# Create module-specific logger for model evaluation activities
logger = logging.getLogger('model_evaluation')    # Unique identifier for this evaluation module
logger.setLevel('DEBUG')                          # Capture all log levels from DEBUG upward
                                                  # Enables detailed debugging and performance monitoring


# Configure real-time console output for immediate feedback during evaluation
console_handler = logging.StreamHandler()         # Handler for terminal/stdout output
console_handler.setLevel('DEBUG')                 # Display all DEBUG and higher messages in console
                                                  # Critical for monitoring evaluation progress


# Configure persistent file logging for audit trails and post-evaluation analysis
log_file_path = os.path.join(log_dir, 'model_evaluation.log')    # Complete path to evaluation log file
file_handler = logging.FileHandler(log_file_path)               # Handler for persistent file storage
file_handler.setLevel('DEBUG')                                  # Store all DEBUG and higher messages
                                                                # Enables retrospective analysis and debugging


# Create standardized log message format with comprehensive metadata
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Format components: ISO_timestamp - module_name - severity_level - descriptive_message
# This format enables automated log parsing and analysis tools

# Apply consistent formatting across both output channels
console_handler.setFormatter(formatter)           # Format real-time console output
file_handler.setFormatter(formatter)             # Format persistent file output


# Register both handlers for comprehensive logging coverage
logger.addHandler(console_handler)               # Enable real-time monitoring
logger.addHandler(file_handler)                 # Enable audit trail generation


# ===== CONFIGURATION PARAMETER LOADING FUNCTION =====
def load_params(params_path: str) -> dict:
    """
    Load experiment configuration parameters from YAML file with validation.
    
    This function provides centralized configuration management for ML evaluation,
    enabling consistent parameter tracking across experiments and reproducible
    evaluation workflows. YAML format supports hierarchical configuration and
    human-readable parameter documentation.
    
    Args:
        params_path (str): File system path to YAML configuration file
                          Expected to contain model evaluation parameters,
                          experiment settings, and metric thresholds
                          Example: 'params.yaml', 'config/evaluation_params.yaml'
        
    Returns:
        dict: Hierarchical dictionary containing all configuration parameters
              Structure enables nested parameter access for different pipeline stages
              Supports parameter versioning and experiment reproducibility
              
    Raises:
        FileNotFoundError: When specified configuration file doesn't exist on disk
        yaml.YAMLError: When YAML file contains syntax errors or malformed structure
        Exception: For any other unexpected file reading or parsing errors
        
    Configuration Structure Example:
        model_evaluation:
          metrics_threshold:
            min_accuracy: 0.85
            min_precision: 0.80
          output_paths:
            metrics_file: 'reports/metrics.json'
    """
    try:
        # Open and parse YAML configuration file with secure loading
        with open(params_path, 'r') as file:
            # Use safe_load to prevent execution of arbitrary code in YAML
            # This is a critical security practice for configuration file handling
            params = yaml.safe_load(file)
            
        # Log successful parameter loading with file path for audit trail
        logger.debug('Parameters retrieved from %s', params_path)
        return params
        
    except FileNotFoundError:
        # Handle missing configuration file with detailed error context
        logger.error('File not found: %s', params_path)
        raise  # Re-raise to halt evaluation with clear error messaging
        
    except yaml.YAMLError as e:
        # Handle YAML parsing errors with specific syntax error details
        logger.error('YAML error: %s', e)
        raise  # Re-raise to provide specific configuration file error context
        
    except Exception as e:
        # Catch unexpected errors during configuration loading
        logger.error('Unexpected error: %s', e)
        raise  # Re-raise to preserve full error context for debugging


# ===== TRAINED MODEL LOADING FUNCTION =====
def load_model(file_path: str):
    """
    Load serialized trained machine learning model from persistent storage.
    
    This function handles the deserialization of trained ML models that were
    saved during the model building phase. It uses pickle format to restore
    the complete model state including learned parameters, hyperparameters,
    and internal algorithm state.
    
    Args:
        file_path (str): Complete file system path to pickled model file
                        Expected to be a binary file created by pickle.dump()
                        Example: './models/random_forest_v1.pkl'
        
    Returns:
        sklearn.base.BaseEstimator: Fully loaded ML model ready for prediction
                                   Preserves all trained parameters and internal state
                                   Can be used immediately for evaluation or inference
    
    Raises:
        FileNotFoundError: When specified model file doesn't exist on disk
        pickle.UnpicklingError: When model file is corrupted or incompatible
        Exception: For any other model loading or deserialization errors
        
    Model Compatibility:
        - Requires same scikit-learn version for proper deserialization
        - Model architecture must match expected input feature dimensions
        - Preserves all hyperparameters and trained weights
    """
    try:
        # Open model file in binary read mode for pickle deserialization
        with open(file_path, 'rb') as file:
            # Deserialize the trained model object from binary format
            # pickle.load restores complete model state including learned parameters
            model = pickle.load(file)
            
        # Log successful model loading with file path for audit tracking
        logger.debug('Model loaded from %s', file_path)
        return model
        
    except FileNotFoundError:
        # Handle missing model file with specific path information
        logger.error('File not found: %s', file_path)
        raise  # Re-raise to halt evaluation with clear file path context
        
    except Exception as e:
        # Handle any other model loading errors (corruption, version mismatch, etc.)
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise  # Re-raise to preserve error context for debugging


# ===== TEST DATA LOADING FUNCTION =====
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed test dataset from CSV file with validation.
    
    This function loads TF-IDF processed test data that was created during
    the feature engineering phase. It provides data integrity validation
    and shape verification for consistent evaluation workflows.
    
    Args:
        file_path (str): Complete path to CSV file containing preprocessed test features
                        Expected format: rows=test_samples, columns=features+labels
                        Example: './data/processed/test_tfidf.csv'
    
    Returns:
        pd.DataFrame: Loaded test dataset with features and ground truth labels
                     Shape: (n_test_samples, n_features + 1)
                     Last column contains true labels for evaluation
    
    Raises:
        pd.errors.ParserError: When CSV file is corrupted or has formatting issues
        FileNotFoundError: When specified file path doesn't exist
        Exception: For any other file reading or data loading errors
        
    Data Validation:
        - Logs dataset shape for verification against expected dimensions
        - Ensures successful CSV parsing with proper delimiter detection
        - Validates data integrity for reliable evaluation metrics
    """
    try:
        # Load CSV file into pandas DataFrame with automatic format detection
        df = pd.read_csv(file_path)
        
        # Log successful data loading with shape information for validation
        logger.debug('Data loaded from %s', file_path)
        return df
        
    except pd.errors.ParserError as e:
        # Handle CSV parsing errors with specific file format context
        logger.error('Failed to parse the CSV file: %s', e)
        raise  # Re-raise to halt evaluation with parsing error details
        
    except Exception as e:
        # Handle any other data loading errors (permissions, disk errors, etc.)
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise  # Re-raise to preserve error information for debugging


# ===== MODEL EVALUATION FUNCTION =====
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Comprehensive model evaluation using multiple classification metrics.
    
    This function performs thorough evaluation of trained classification models
    using industry-standard metrics. It supports both hard predictions and
    probability scores for comprehensive performance assessment.
    
    Args:
        clf: Trained classifier object with predict() and predict_proba() methods
             Must be compatible with scikit-learn estimator interface
        X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features)
                            Contains TF-IDF vectors for evaluation samples
        y_test (np.ndarray): Ground truth labels of shape (n_samples,)
                            Contains true class labels for performance comparison
    
    Returns:
        dict: Comprehensive metrics dictionary containing:
              - accuracy: Overall classification accuracy (0.0 to 1.0)
              - precision: Positive class precision (quality of positive predictions)
              - recall: Positive class recall (coverage of actual positives)
              - auc: Area under ROC curve (discrimination ability)
    
    Raises:
        ValueError: When input arrays have incompatible shapes or invalid data
        AttributeError: When model doesn't support required prediction methods
        Exception: For any other evaluation computation errors
        
    Evaluation Metrics Details:
        - Accuracy: (TP + TN) / (TP + TN + FP + FN) - Overall correctness
        - Precision: TP / (TP + FP) - Quality of positive predictions
        - Recall: TP / (TP + FN) - Coverage of actual positive cases
        - AUC: Area under ROC curve - Discrimination threshold independence
    """
    try:
        # Generate hard predictions using trained model's decision boundary
        # predict() returns class labels based on learned decision threshold
        y_pred = clf.predict(X_test)
        
        # Generate probability predictions for positive class (index 1)
        # predict_proba() returns class probabilities for ROC analysis
        # [:, 1] extracts positive class probabilities for AUC calculation
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate comprehensive classification metrics for model assessment
        
        # Overall accuracy: fraction of correctly predicted samples
        accuracy = accuracy_score(y_test, y_pred)
        
        # Precision: quality of positive predictions (avoid false alarms)
        precision = precision_score(y_test, y_pred)
        
        # Recall: coverage of actual positive samples (avoid missing positives)
        recall = recall_score(y_test, y_pred)
        
        # AUC: discrimination ability independent of classification threshold
        auc = roc_auc_score(y_test, y_pred_proba)

        # Organize metrics in structured dictionary for easy access and persistence
        metrics_dict = {
            'accuracy': accuracy,      # Overall prediction correctness
            'precision': precision,    # Positive prediction quality
            'recall': recall,         # Positive case coverage
            'auc': auc               # Threshold-independent discrimination
        }
        
        # Log successful metrics calculation for monitoring and debugging
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
        
    except Exception as e:
        # Handle any evaluation computation errors with detailed context
        logger.error('Error during model evaluation: %s', e)
        raise  # Re-raise to preserve error information for debugging


# ===== METRICS PERSISTENCE FUNCTION =====
def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Persist evaluation metrics to JSON file for reporting and comparison.
    
    This function saves computed evaluation metrics in structured JSON format,
    enabling easy access for reporting dashboards, model comparison, and
    automated decision-making in MLOps pipelines.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics and scores
                       Expected to contain numeric values for various performance measures
                       Example: {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88}
        file_path (str): Complete file system path for JSON output file
                        Example: 'reports/model_v1_metrics.json'
                        
    Returns:
        None: Function performs file I/O operations without return value
        
    Raises:
        FileNotFoundError: When specified directory path doesn't exist
        json.JSONEncodeError: When metrics contain non-serializable data types
        Exception: For any other file writing or serialization errors
        
    Output Format:
        - Human-readable JSON with 4-space indentation
        - Structured format compatible with reporting tools
        - Preserves metric precision for accurate comparison
    """
    try:
        # Ensure target directory exists before attempting file write
        # os.path.dirname() extracts directory path from complete file path
        # makedirs() creates directory structure if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write metrics dictionary to JSON file with readable formatting
        with open(file_path, 'w') as file:
            # json.dump() serializes dictionary to JSON with pretty printing
            # indent=4 creates human-readable format with proper spacing
            json.dump(metrics, file, indent=4)
            
        # Log successful metrics persistence with file location
        logger.debug('Metrics saved to %s', file_path)
        
    except Exception as e:
        # Handle any file writing or JSON serialization errors
        logger.error('Error occurred while saving the metrics: %s', e)
        raise  # Re-raise to preserve error context for debugging


# ===== MAIN EVALUATION ORCHESTRATION FUNCTION =====
def main():
    """
    Main orchestration function for complete model evaluation workflow with MLOps integration.
    
    This function coordinates all phases of model evaluation from configuration loading
    through metrics computation and experiment tracking. It integrates with DVC Live
    for comprehensive experiment management and reproducibility.
    
    Evaluation Pipeline Stages:
    1. Load evaluation configuration parameters from YAML
    2. Load trained model from persistent storage
    3. Load preprocessed test dataset with features and labels
    4. Prepare feature matrix and target vector for evaluation
    5. Compute comprehensive evaluation metrics
    6. Track experiment with DVC Live for reproducibility
    7. Persist metrics to JSON for reporting and comparison
    
    MLOps Integration:
        - DVC Live experiment tracking for version control
        - Parameter logging for reproducible experiments
        - Metric persistence for automated model selection
        - Comprehensive audit trails through logging
        
    File Structure Requirements:
        params.yaml                           # Configuration parameters
        ./models/model.pkl                   # Trained model file
        ./data/processed/test_tfidf.csv     # Preprocessed test features
        reports/metrics.json                 # Output metrics file
    """
    try:
        # ===== STAGE 1: CONFIGURATION LOADING =====
        # Load evaluation parameters and experiment configuration
        params = load_params(params_path='params.yaml')
        
        # ===== STAGE 2: MODEL LOADING =====
        # Load trained model from persistent storage for evaluation
        clf = load_model('./models/model.pkl')
        
        # ===== STAGE 3: TEST DATA LOADING =====
        # Load preprocessed test dataset with TF-IDF features
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        # ===== STAGE 4: DATA PREPARATION =====
        # Prepare evaluation data by separating features from labels
        # iloc[:, :-1] selects all columns except last (feature matrix)
        # iloc[:, -1] selects only last column (ground truth labels)
        X_test = test_data.iloc[:, :-1].values    # Test feature matrix: (n_samples, n_features)
        y_test = test_data.iloc[:, -1].values     # Ground truth labels: (n_samples,)

        # ===== STAGE 5: MODEL EVALUATION =====
        # Compute comprehensive evaluation metrics using test data
        metrics = evaluate_model(clf, X_test, y_test)

        # ===== STAGE 6: EXPERIMENT TRACKING WITH DVC LIVE =====
        # Initialize DVC Live context for experiment tracking and versioning
        # save_dvc_exp=True enables automatic experiment versioning
        with Live(save_dvc_exp=True) as live:
            #
            # The metrics are being computed using y_test vs y_pred
            # This would result in scores
            # The  implementation use the actual predictions:
            
            # Get predictions for proper metric calculation
            y_pred = clf.predict(X_test)
            
            # Log actual evaluation metrics to DVC Live experiment tracker
            live.log_metric('accuracy', accuracy_score(y_test, y_pred))
            live.log_metric('precision', precision_score(y_test, y_pred))
            live.log_metric('recall', recall_score(y_test, y_pred))
            
            # Log experiment parameters for reproducibility
            live.log_params(params)
        
        # ===== STAGE 7: METRICS PERSISTENCE =====
        # Save evaluation metrics to JSON file for reporting and comparison
        save_metrics(metrics, 'reports/metrics.json')
        
        # Log successful completion of evaluation pipeline
        print("✅ Model evaluation completed successfully!")
        print(f"📊 Metrics saved to: reports/metrics.json")
        print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall: {metrics['recall']:.4f}")
        print(f"📈 AUC: {metrics['auc']:.4f}")
        
    except Exception as e:
        # ===== COMPREHENSIVE ERROR HANDLING =====
        # Handle any errors during the complete evaluation process
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"❌ Error: {e}")
        print("🔍 Check logs for detailed error information")


# ===== SCRIPT ENTRY POINT =====
# Ensures main() function only executes when script is run directly,
# not when imported as a module in other components of the ML pipeline
if __name__ == '__main__':
    # Execute complete model evaluation pipeline with MLOps integration
    main()

# import os
# import numpy as np
# import pandas as pd
# import pickle
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import logging
# import yaml
# from dvclive import Live

# # Ensure the "logs" directory exists
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# # logging configuration
# logger = logging.getLogger('model_evaluation')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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

# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', file_path)
#         return model
#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the model: %s', e)
#         raise

# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
#     """Evaluate the model and return the evaluation metrics."""
#     try:
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_pred_proba)

#         metrics_dict = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'auc': auc
#         }
#         logger.debug('Model evaluation metrics calculated')
#         return metrics_dict
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise

# def save_metrics(metrics: dict, file_path: str) -> None:
#     """Save the evaluation metrics to a JSON file."""
#     try:
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)

#         with open(file_path, 'w') as file:
#             json.dump(metrics, file, indent=4)
#         logger.debug('Metrics saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the metrics: %s', e)
#         raise

# def main():
#     try:
#         params = load_params(params_path='params.yaml')
#         clf = load_model('./models/model.pkl')
#         test_data = load_data('./data/processed/test_tfidf.csv')
        
#         X_test = test_data.iloc[:, :-1].values
#         y_test = test_data.iloc[:, -1].values

#         metrics = evaluate_model(clf, X_test, y_test)

#         # Experiment tracking using dvclive
#         with Live(save_dvc_exp=True) as live:
#             live.log_metric('accuracy', accuracy_score(y_test, y_test))
#             live.log_metric('precision', precision_score(y_test, y_test))
#             live.log_metric('recall', recall_score(y_test, y_test))

#             live.log_params(params)
        
#         save_metrics(metrics, 'reports/metrics.json')
#     except Exception as e:
#         logger.error('Failed to complete the model evaluation process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()