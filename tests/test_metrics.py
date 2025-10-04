import numpy as np
from src.metrics import classification_metrics

def test_classification_metrics():
    """
    Test classification metrics calculation.
    """
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    
    metrics = classification_metrics(y_true, y_pred)
    
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= v <= 1 for v in metrics.values())
    assert metrics["accuracy"] == 5/6  # 5 correct out of 6

def test_perfect_classification():
    """
    Test metrics with perfect predictions.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    
    metrics = classification_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
