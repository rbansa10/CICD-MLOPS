"""
Data Preprocessing Pipeline for Text Classification
==================================================
This script performs comprehensive text preprocessing for machine learning models including:
- Text normalization (lowercase, tokenization, stemming)
- Label encoding for target variables
- Duplicate removal and data cleaning
- Interactive file selection using GUI
- Robust logging and error handling

Author: Rajat Bansal
Date: September 29, 2025
Version: 2.0 (GUI-enabled file selection)
"""

# ===== IMPORT STATEMENTS =====
# Standard library imports for system operations and GUI functionality
import os                           # Operating system interface for file/directory operations
import logging                     # Comprehensive logging system for debugging and monitoring
import pandas as pd                # Data manipulation and analysis library

# Machine learning preprocessing utilities
from sklearn.preprocessing import LabelEncoder  # Converts categorical labels to numerical format

# Natural Language Processing (NLP) libraries from NLTK
from nltk.stem.porter import PorterStemmer     # Reduces words to their root/stem form
from nltk.corpus import stopwords              # Common words to remove (the, and, is, etc.)
import string                                  # String constants and utilities for punctuation
import nltk                                    # Natural Language Toolkit main library

# GUI libraries for interactive file selection
import tkinter as tk                           # Main GUI framework
from tkinter import filedialog                # File dialog boxes for user interaction

# ===== NLTK DATA DOWNLOADS =====
# Download required NLTK datasets for text processing
# These downloads happen once and are cached locally for future use
nltk.download('stopwords')    # Download list of common English stopwords
nltk.download('punkt')        # Download tokenization models for sentence/word splitting


# ===== LOGGING DIRECTORY SETUP =====
# Create directory structure for storing log files
# This ensures the application can write logs without FileNotFoundError
log_dir = 'logs'                              # Directory name for log files
os.makedirs(log_dir, exist_ok=True)          # Create directory if it doesn't exist
                                             # exist_ok=True prevents error if already exists


# ===== COMPREHENSIVE LOGGING CONFIGURATION =====
# Set up dual logging system (console + file) for complete monitoring
# This allows real-time debugging and permanent log storage

# Create named logger instance for this specific module
logger = logging.getLogger('data_preprocessing')  # Module-specific logger name
logger.setLevel('DEBUG')                          # Capture all log levels from DEBUG up


# Configure console handler for real-time terminal output
console_handler = logging.StreamHandler()         # Handler for terminal/console output
console_handler.setLevel('DEBUG')                 # Show DEBUG level and above in console


# Configure file handler for permanent log storage
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')  # Full path to log file
file_handler = logging.FileHandler(log_file_path)               # Handler for file output
file_handler.setLevel('DEBUG')                                  # Store DEBUG level and above in file


# Create standardized log message format with timestamp and context
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Format components: timestamp - logger_name - log_level - actual_message

# Apply formatting to both handlers
console_handler.setFormatter(formatter)           # Format console messages
file_handler.setFormatter(formatter)             # Format file messages


# Register handlers with the logger to enable dual output
logger.addHandler(console_handler)               # Enable console logging
logger.addHandler(file_handler)                 # Enable file logging


# ===== TEXT TRANSFORMATION FUNCTION =====
def transform_text(text):
    """
    Comprehensive text preprocessing pipeline for NLP tasks.
    
    This function applies multiple text normalization techniques to prepare raw text
    for machine learning models. The preprocessing steps ensure consistent text format
    and remove noise that could negatively impact model performance.
    
    Processing Steps:
    1. Convert to lowercase for case-insensitive processing
    2. Tokenize into individual words using NLTK
    3. Remove non-alphanumeric tokens (special characters, numbers-only tokens)
    4. Filter out English stopwords (common words like 'the', 'and', 'is')
    5. Remove punctuation marks
    6. Apply Porter stemming to reduce words to root forms
    7. Rejoin tokens into clean text string
    
    Args:
        text (str): Raw input text to be processed
        
    Returns:
        str: Cleaned and normalized text ready for vectorization
        
    Example:
        Input: "The running dogs are quickly jumping!"
        Output: "run dog quick jump"
    """
    # Initialize Porter Stemmer for word root extraction
    # Porter Stemmer reduces words to their linguistic roots (running -> run, better -> better)
    ps = PorterStemmer()
    
    # Step 1: Convert entire text to lowercase for case-insensitive processing
    # This ensures "The" and "the" are treated as the same word
    text = text.lower()
    
    # Step 2: Tokenize text into individual words using NLTK's word tokenizer
    # This splits text on whitespace and punctuation boundaries
    # Example: "hello world!" -> ["hello", "world", "!"]
    text = nltk.word_tokenize(text)
    
    # Step 3: Filter tokens to keep only alphanumeric words
    # This removes pure punctuation tokens, pure numbers, and special characters
    # isalnum() returns True only for tokens containing letters and/or numbers
    text = [word for word in text if word.isalnum()]
    
    # Step 4: Remove English stopwords and punctuation
    # Stopwords are common words that don't carry significant meaning for classification
    # string.punctuation contains all standard punctuation marks
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Step 5: Apply Porter stemming to reduce words to their root forms
    # This normalizes different word forms: running/runs/ran -> run
    # Stemming helps the model recognize semantically similar words
    text = [ps.stem(word) for word in text]
    
    # Step 6: Join processed tokens back into a single clean string
    # Space-separated words ready for vectorization by TF-IDF or other methods
    return " ".join(text)


# ===== DATAFRAME PREPROCESSING FUNCTION =====
def preprocess_df(df, text_column='text', target_column='target'):
    """
    Comprehensive DataFrame preprocessing pipeline for machine learning.
    
    This function applies multiple preprocessing steps to prepare a dataset
    for machine learning model training. It handles both text data normalization
    and target variable encoding while maintaining data integrity.
    
    Processing Steps:
    1. Label encoding of target variable (categorical -> numerical)
    2. Duplicate row removal for data quality
    3. Text column transformation using NLP preprocessing
    
    Args:
        df (pd.DataFrame): Input DataFrame containing text and target columns
        text_column (str): Name of column containing text data (default: 'text')
        target_column (str): Name of column containing target labels (default: 'target')
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with encoded targets and cleaned text
        
    Raises:
        KeyError: If specified columns don't exist in the DataFrame
        Exception: For any other preprocessing errors
    """
    try:
        # Log the start of preprocessing for debugging and monitoring
        logger.debug('Starting preprocessing for DataFrame')
        
        # Step 1: Encode target variable from categorical to numerical format
        # LabelEncoder converts string/categorical labels to integers (0, 1, 2, ...)
        # This is required for machine learning algorithms that need numerical targets
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')  # Log successful encoding

        # Step 2: Remove duplicate rows to improve data quality
        # keep='first' retains the first occurrence of duplicate rows
        # This prevents the model from being biased by repeated samples
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')  # Log successful deduplication
        
        # Step 3: Apply comprehensive text transformation to the text column
        # .loc[:, text_column] ensures we're modifying the DataFrame properly
        # apply() function applies transform_text to each row in the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')  # Log successful text processing
        
        return df  # Return the fully preprocessed DataFrame
    
    except KeyError as e:
        # Handle missing column errors with specific error message
        logger.error('Column not found: %s', e)
        raise  # Re-raise to stop execution with informative error
        
    except Exception as e:
        # Handle any other preprocessing errors with generic handling
        logger.error('Error during text normalization: %s', e)
        raise  # Re-raise to stop execution and preserve error information


# ===== INTERACTIVE FILE SELECTION FUNCTION =====
def pick_file(title="Select CSV file"):
    """
    Interactive GUI file picker for CSV file selection.
    
    This function creates a file dialog window that allows users to visually
    select CSV files instead of hardcoding file paths. This makes the script
    more flexible and user-friendly for different datasets.
    
    Args:
        title (str): Dialog window title text (default: "Select CSV file")
        
    Returns:
        str: Full path to the selected CSV file, or empty string if cancelled
        
    Technical Details:
    - Creates temporary Tkinter root window for dialog functionality
    - Filters for CSV files but allows "All files" as fallback
    - Properly destroys GUI resources after selection
    """
    # Create root Tkinter window (required for dialog functionality)
    root = tk.Tk()
    
    # Hide the main window since we only need the dialog
    # withdraw() makes the window invisible while keeping it functional
    root.withdraw()
    
    # Open file selection dialog with CSV file filtering
    file_path = filedialog.askopenfilename(
        title=title,                                          # Dialog window title
        filetypes=[("CSV files", "*.csv"),                   # Primary filter: CSV files
                  ("All files", "*.*")]                      # Fallback: all file types
    )
    
    # Clean up GUI resources to prevent memory leaks
    root.destroy()
    
    return file_path  # Return selected file path (empty string if cancelled)


# ===== MAIN EXECUTION FUNCTION =====
def main(text_column='text', target_column='target'):
    """
    Main orchestration function for the data preprocessing pipeline.
    
    This function coordinates the entire preprocessing workflow from raw data
    input to processed data output. It includes interactive file selection,
    comprehensive error handling, and organized data storage.
    
    Workflow:
    1. Interactive GUI selection of train and test CSV files
    2. File validation and existence checking
    3. Data loading with pandas
    4. Comprehensive preprocessing (text + targets)
    5. Organized storage in interim data directory
    
    Args:
        text_column (str): Name of text column in datasets (default: 'text')
        target_column (str): Name of target column in datasets (default: 'target')
        
    File Structure Created:
        ./data/interim/train_processed.csv  # Processed training data
        ./data/interim/test_processed.csv   # Processed test data
    """
    try:
        # ===== INTERACTIVE FILE SELECTION =====
        # Use GUI dialogs for user-friendly file selection instead of hardcoded paths
        
        # Select training dataset file
        print("Pick the TRAIN CSV file:")                    # User instruction
        train_file = pick_file("Select TRAIN CSV file")      # GUI file picker
        print(f"TRAIN file selected: {train_file}")          # Confirmation message
        
        # Select test dataset file  
        print("Pick the TEST CSV file:")                     # User instruction
        test_file = pick_file("Select TEST CSV file")        # GUI file picker  
        print(f"TEST file selected: {test_file}")            # Confirmation message
        
        # ===== FILE VALIDATION =====
        # Verify that files were selected and actually exist on disk
        
        # Validate training file selection and existence
        if not train_file or not os.path.isfile(train_file):
            error_msg = f'Required file not found: {train_file}'
            logger.error(error_msg)                          # Log the error
            print(f"Error: {error_msg}")                     # User-friendly error message
            return                                           # Exit function early
            
        # Validate test file selection and existence    
        if not test_file or not os.path.isfile(test_file):
            error_msg = f'Required file not found: {test_file}'
            logger.error(error_msg)                          # Log the error
            print(f"Error: {error_msg}")                     # User-friendly error message
            return                                           # Exit function early

        # ===== DATA LOADING =====
        # Load CSV files into pandas DataFrames for processing
        train_data = pd.read_csv(train_file)                 # Load training dataset
        test_data = pd.read_csv(test_file)                   # Load test dataset
        logger.debug('Data loaded properly')                 # Log successful loading

        # ===== DATA PREPROCESSING =====
        # Apply comprehensive preprocessing pipeline to both datasets
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # ===== PROCESSED DATA STORAGE =====
        # Create organized directory structure for interim processed data
        data_path = os.path.join("./data", "interim")        # Interim data directory path
        os.makedirs(data_path, exist_ok=True)               # Create directory if needed
        
        # Save processed datasets with descriptive filenames
        train_output_path = os.path.join(data_path, "train_processed.csv")
        test_output_path = os.path.join(data_path, "test_processed.csv")
        
        # Write processed DataFrames to CSV files (index=False to avoid extra index column)
        train_processed_data.to_csv(train_output_path, index=False)
        test_processed_data.to_csv(test_output_path, index=False)
        
        # Log successful completion with output location
        logger.debug('Processed data saved to %s', data_path)
        print(f"✅ Preprocessing completed successfully!")
        print(f"📁 Processed files saved to: {data_path}")
        
    # ===== COMPREHENSIVE ERROR HANDLING =====
    except FileNotFoundError as e:
        # Handle missing file errors specifically
        error_msg = f'File not found: {e}'
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        
    except pd.errors.EmptyDataError as e:
        # Handle empty CSV file errors
        error_msg = f'No data in file: {e}'
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        
    except Exception as e:
        # Handle any other unexpected errors with full error information
        error_msg = f'Failed to complete the data transformation process: {e}'
        logger.error(error_msg)
        print(f"❌ Error: {e}")


# ===== COMMENTED OUT ALTERNATIVE MAIN FUNCTION =====
# This is the original hardcoded version kept for reference and fallback
# Uncomment this version if you prefer fixed file paths instead of GUI selection

# def main(text_column='text', target_column='target'):
#     """
#     Alternative main function with hardcoded file paths (LEGACY VERSION).
#     
#     This version assumes fixed file structure:
#     - Training data: ./data/raw/train.csv  
#     - Test data: ./data/raw/test.csv
#     
#     Use this version for automated pipelines where GUI interaction isn't desired.
#     """
#     try:
#         # Load data from fixed file paths
#         train_data = pd.read_csv('./data/raw/train.csv')
#         test_data = pd.read_csv('./data/raw/test.csv')
#         logger.debug('Data loaded properly')
# 
#         # Apply preprocessing pipeline
#         train_processed_data = preprocess_df(train_data, text_column, target_column)
#         test_processed_data = preprocess_df(test_data, text_column, target_column)
# 
#         # Save processed data to interim directory
#         data_path = os.path.join("./data", "interim")
#         os.makedirs(data_path, exist_ok=True)
#         
#         train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
#         test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
#         
#         logger.debug('Processed data saved to %s', data_path)
#         
#     except FileNotFoundError as e:
#         logger.error('File not found: %s', e)
#     except pd.errors.EmptyDataError as e:
#         logger.error('No data: %s', e)
#     except Exception as e:
#         logger.error('Failed to complete the data transformation process: %s', e)
#         print(f"Error: {e}")


# ===== SCRIPT ENTRY POINT =====
# This ensures the main() function only executes when the script is run directly,
# not when imported as a module by other scripts
if __name__ == '__main__':
    # Execute the preprocessing pipeline with default column names
    # Modify these parameters if your CSV files use different column names
    main()  # Uses default: text_column='text', target_column='target'
