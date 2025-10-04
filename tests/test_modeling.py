import pandas as pd
import numpy as np
from src.modeling import ModelBuilder

def test_model_builder_init():
    """
    Test ModelBuilder initialization.
    """
    params = {"max_iter": 1000, "solver": "liblinear"}
    builder = ModelBuilder(random_state=42, params=params)
    
    assert builder.random_state == 42
    assert builder.params == params
    assert hasattr(builder, "model")

def test_model_builder_invalid_params():
    """
    Test ModelBuilder with invalid parameters.
    """
    try:
        ModelBuilder(random_state=42, params={"invalid_param": "value"})
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def test_model_training_and_prediction():
    """
    Test model training and prediction.
    """
    X_train = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame([[2, 3], [6, 7]])
    
    builder = ModelBuilder(random_state=42, params={"max_iter": 1000})
    trained_model = builder.train_model(X_train, y_train)
    predictions = builder.predict(X_test)
    
    assert trained_model is not None
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)
