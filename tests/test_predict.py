import os
import tempfile
import pandas as pd
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch, MagicMock
from src.io_utils import save_model

def test_predict_main_with_model():
    """
    Test predict main function with existing model.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and save a test model
        model = LogisticRegression(random_state=42)
        X_train = [[1, 2, 3, 0, 0.5, 6, 50, 4, 1, 300, 15, 390, 5],
                   [2, 0, 5, 1, 0.6, 7, 60, 3, 2, 400, 16, 380, 10]]
        y_train = [0, 1]
        model.fit(X_train, y_train)
        
        model_path = os.path.join(temp_dir, "model.joblib")
        save_model(model, model_path)
        
        # Mock sys.argv to simulate command line arguments
        test_args = ["predict.py", "--model_path", model_path, 
                    "--output_csv", os.path.join(temp_dir, "predictions.csv")]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                from src.predict import main
                main()
                
                # Check that predictions were printed
                mock_print.assert_called()
                
                # Check that predictions file was created
                pred_file = os.path.join(temp_dir, "predictions.csv")
                assert os.path.exists(pred_file)

def test_predict_with_sample_data():
    """
    Test that default sample data has correct structure.
    """
    # Default sample data from predict.py
    default_data = pd.DataFrame({"CRIM": [0.03, 0.1],
                                "ZN": [18.0, 0.0],
                                "INDUS": [2.31, 7.07],
                                "CHAS": [0, 0],
                                "NOX": [0.538, 0.469],
                                "RM": [6.575, 6.0],
                                "AGE": [65.2, 68.2],
                                "DIS": [4.09, 3.5],
                                "RAD": [1, 2],
                                "TAX": [296, 242],
                                "PTRATIO": [15.3, 17.8],
                                "B": [396.9, 392.8],
                                "LSTAT": [4.98, 9.14]})
    
    assert len(default_data) == 2
    assert len(default_data.columns) == 13
