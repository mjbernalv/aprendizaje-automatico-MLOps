from __future__ import annotations
from sklearn.linear_model import LogisticRegression
import pandas as pd

class ModelBuilder:
    def __init__(self, random_state: int, params: dict):
        self.random_state = random_state
        self.params = params
        try:
            self.model = LogisticRegression(random_state = self.random_state, **self.params)
        except Exception as e:
            raise ValueError(f"Error al construir el modelo: {e}")


    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """"
        Trains the model with the provided training data.
        
        :param X_train: Training features
        :param y_train: Training target
        :return: Trained model
        """
        self.model.fit(X_train, y_train)
        return self.model


    def predict(self, X):
        """"
        Makes predictions using the trained model.

        :param X: Input features for prediction
        :return y_pred: Predicted values
        """
        y_pred = self.model.predict(X)
        return y_pred
