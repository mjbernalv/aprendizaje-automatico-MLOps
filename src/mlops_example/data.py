from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from typing import Tuple
import pandas as pd
import warnings


def load_boston_as_regression(seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads Boston Housing as a regression problem with y = MEDV.

    :param seed: random seed for reproducibility
    :return: Tuple of features (X) and target (y)
    :raises RuntimeError: If the dataset cannot be loaded
    """
    try:    
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        X = boston.data
        y = boston.target.astype(float)
        return X, y
    except Exception as e:
        raise RuntimeError("No se pudo cargar Boston Housing desde OpenML.") from e


def load_boston_as_classification(seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads Boston Housing as a binary classification problem: 1 if MEDV > median, 0 otherwise.
    
    :param seed: random seed for reproducibility
    :return: Tuple of features (X) and binary target (y)
    """
    X, y_reg = load_boston_as_regression(seed)
    threshold = y_reg.median()
    y = (y_reg > threshold).astype(int)
    
    return X, y


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