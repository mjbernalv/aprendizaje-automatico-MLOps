from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Literal, Tuple

ModelType = Literal["rf", "logreg"]

def build_model(model_type: ModelType, **params):
    """
    Creates and returns a machine learning model based on the specified type and parameters.

    :param model_type: Type of model to create ("rf" for RandomForestRegressor, "logreg" for LogisticRegression)
    :param params: Additional parameters to pass to the model constructor
    :return: An instance of the specified model
    :raises ValueError: If an unsupported model type is provided
    """
    if model_type == "rf":
        return RandomForestRegressor(random_state = 42, **params)
    elif model_type == "logreg":
        return LogisticRegression(max_iter = 1000, solver = "liblinear", random_state = 42, **params)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")