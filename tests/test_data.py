import pandas as pd
import numpy as np
from src.data import load_data, train_test_split_xy, normalize_data, remove_outliers

def test_load_data():
    """
    Test loading Boston Housing as binary classification.
    """
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert X.shape[0] > 100
    assert set(y.unique()) == {0, 1}


def test_train_test_split_xy():
    """
    Test train-test split functionality.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size=0.3, seed=42)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert abs(len(X_test) / len(X) - 0.3) < 0.05


def test_normalize_data():
    """
    Test data normalization using StandardScaler.
    """
    X, _ = load_data()
    X_norm = normalize_data(X)
    
    assert X_norm.shape == X.shape
    assert list(X_norm.columns) == list(X.columns)
    assert np.allclose(X_norm.mean().values, 0, atol=1e-10)
    # Use ddof=0 for population standard deviation (matches StandardScaler)
    assert np.allclose(X_norm.std(ddof=0).values, 1, atol=1e-10)


def test_remove_outliers():
    """
    Test outlier removal functionality.
    """
    X, y = load_data()
    # Ensure X is numeric - convert object columns to float
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        # If there are non-numeric columns, convert them
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X_clean, y_clean = remove_outliers(X, y, contamination=0.1)
    
    assert len(X_clean) == len(y_clean)
    assert len(X_clean) < len(X)
    assert len(X_clean) >= int(0.8 * len(X))