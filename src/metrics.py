from __future__ import annotations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """"
    Calculates classification metric: Accuracy, Precision, Recall, and F1 Score.

    :param y_true: True target values
    :param y_pred: Predicted target values
    :return: Dictionary with the calculated metrics
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}