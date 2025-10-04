from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import fetch_openml
from typing import Tuple
import pandas as pd
import warnings


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads Boston Housing as a binary classification problem: 1 if MEDV > median, 0 otherwise.

    :param seed: random seed for reproducibility
    :return: Tuple of features (X) and target (y)
    :raises RuntimeError: If the dataset cannot be loaded
    """
    try:    
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        X = boston.data
        y_reg = boston.target.astype(float)
        y = (y_reg > y_reg.median()).astype(int)
        return X, y
    except Exception as e:
        raise RuntimeError("No se pudo cargar Boston Housing desde OpenML.") from e


def train_test_split_xy(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """
    Splits features and target into training and testing sets.

    :param X: Features DataFrame
    :param y: Target Series
    :param test_size: Proportion of the dataset to include in the test split
    :param seed: Random seed for reproducibility
    :return: Tuple of X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed, shuffle = True)

    return X_train, X_test, y_train, y_test


def normalize_data(X: pd.DataFrame) -> pd.DataFrame:
    """"
    Normalizes data using the Standard Scaler.

    :param X: Data to be normalized
    :return X_norm: Normalized data
    """

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns = X.columns, index = X.index)
    
    return X_norm 


def remove_outliers(X: pd.DataFrame, y: pd.Series, contamination: float = 0.05) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Removes outliers from the dataset using the Elliptic Envelope method.

    :param X: Features DataFrame
    :param y: Target Series
    :return: Tuple of cleaned features (X_clean) and target (y_clean)
    """

    ee = EllipticEnvelope(contamination = contamination, random_state = 42)
    preds = ee.fit_predict(X)
    
    mask = preds != -1
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean