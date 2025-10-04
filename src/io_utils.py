from __future__ import annotations
import pandas as pd
import joblib
import time
import os


def timestamped_dir(base_dir: str) -> str:
    """"
    Creates a timestamped directory inside the specified base directory.

    :param base_dir: Base directory where the timestamped directory will be created
    :return: Path to the created timestamped directory
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, ts)
    os.makedirs(path, exist_ok = True)

    return path


def ensure_latest_symlink(base_dir: str, latest_name: str = "latest", target_dir: str | None = None) -> str:
    """"
    Ensures a symlink named `latest_name` points to the most recent directory in `base_dir`.

    :param base_dir: Base directory containing timestamped directories
    :param latest_name: Name of the symlink to create or update
    :param target_dir: Specific directory to point the symlink to; if None, points to the latest
    :return: Path to the symlink
    """
    latest_path = os.path.join(base_dir, latest_name)
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        try:
            os.remove(latest_path)
        except OSError:
            pass
    if target_dir is None:
        candidates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if not candidates:
            return latest_path
        target_dir = os.path.join(base_dir, candidates[-1])
        os.symlink(os.path.abspath(target_dir), latest_path)
        return latest_path


def save_model(model, path: str):
    """"
    Saves a machine learning model to the specified path using joblib.
    
    :param model: The machine learning model to save
    :param path: The file path where the model will be saved
    """
    joblib.dump(model, path)


def load_model(path: str):
    """"
    Loads a machine learning model from the specified path using joblib.

    :param path: The file path from which the model will be loaded
    :return: The loaded machine learning model
    """
    return joblib.load(path)


def save_predictions(preds, path_csv: str, index: bool = False):
    """
    Saves the model predictions to a CSV file.

    :param preds: The predictions to save
    :param path_csv: The file path where the predictions will be saved
    :param index: Whether to include the index in the CSV file
    """
    df = pd.DataFrame({"prediction": preds})
    df.to_csv(path_csv, index = index)