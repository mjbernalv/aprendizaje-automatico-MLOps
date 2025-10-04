import os
import tempfile
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.io_utils import timestamped_dir, save_model, load_model, save_predictions

def test_timestamped_dir():
    """
    Test timestamped directory creation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        path = timestamped_dir(temp_dir)
        assert os.path.exists(path)
        assert os.path.basename(path).replace("_", "").isdigit()

def test_save_load_model():
    """
    Test model saving and loading.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and train a simple model
        model = LogisticRegression(random_state=42)
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        model.fit(X, y)
        
        # Save and reload
        model_path = os.path.join(temp_dir, "model.joblib")
        save_model(model, model_path)
        loaded_model = load_model(model_path)
        
        # Test predictions are the same
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)
        assert all(pred_original == pred_loaded)

def test_save_predictions():
    """
    Test predictions saving.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        preds = [0, 1, 1, 0]
        csv_path = os.path.join(temp_dir, "preds.csv")
        
        save_predictions(preds, csv_path)
        
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert list(df["prediction"]) == preds
