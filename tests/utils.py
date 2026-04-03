import os
import json
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from datetime import datetime

def setup_logging(log_file: str = 'application.log', level: int = logging.INFO) -> logging.Logger:
    """
    Sets up logging configuration for the application.

    Args:
        log_file (str): The name of the log file. Defaults to 'application.log'.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration file from the given path.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The configuration as a dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file: {config_path}")

def save_model(model, model_path: str) -> None:
    """
    Saves the trained model to the specified path.

    Args:
        model: The trained machine learning model to save.
        model_path (str): The path where the model should be saved.
    """
    import pickle
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model to {model_path}: {e}")

def load_model(model_path: str):
    """
    Loads a trained model from the specified path.

    Args:
        model_path (str): The path to the saved model.

    Returns:
        The loaded machine learning model.
    """
    import pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The random seed to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Add more seed setting for other libraries if needed, e.g., TensorFlow, PyTorch
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)

def create_directory(path: str) -> None:
    """
    Creates a directory if it doesn't exist.

    Args:
        path (str): The path to the directory.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        raise OSError(f"Error creating directory {path}: {e}")

def current_timestamp() -> str:
    """
    Returns the current timestamp in a string format.

    Returns:
        str: The current timestamp.
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates some common metrics for classification tasks.

    Args:
        y_true (np.ndarray): The ground truth labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics