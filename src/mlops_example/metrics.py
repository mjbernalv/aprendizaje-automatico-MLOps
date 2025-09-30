from __future__ import annotations
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """"
    Calculates regression metrics: RMSE, R², MAE, and MAPE.

    :param y_true: True target values
    :param y_pred: Predicted target values
    :return: Dictionary with RMSE, R², MAE, and MAPE values
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"rmse": rmse, "r2": r2, "mae": mae, "mape": mape}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """"
    Calculates classification metric: Accuracy, Precision, Recall, and F1 Score.

    :param y_true: True target values
    :param y_pred: Predicted target values
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}