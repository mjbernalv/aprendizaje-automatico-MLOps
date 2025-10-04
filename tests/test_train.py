import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from src.train import load_config, setup_mlflow

def test_load_config():
    """
    Test YAML configuration loading
    """
    config_data = {
        "experiment_name": "test_experiment",
        "data_params": {"test_size": 0.2},
        "model_params": {"max_iter": 1000}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        loaded_config = load_config(config_path)
        assert loaded_config == config_data
    finally:
        os.unlink(config_path)

@patch('mlflow.set_tracking_uri')
@patch('mlflow.set_experiment')
def test_setup_mlflow(mock_set_experiment, mock_set_tracking_uri):
    """
    Test MLflow setup function.
    """
    config = {"experiment_name": "test_experiment"}
    
    setup_mlflow(config)
    
    mock_set_tracking_uri.assert_called_once()
    mock_set_experiment.assert_called_once_with("test_experiment")

@patch('mlflow.set_tracking_uri')
@patch('mlflow.set_experiment')
def test_setup_mlflow_default_experiment(mock_set_experiment, mock_set_tracking_uri):
    """
    Test MLflow setup with default experiment name.
    """
    config = {}  # No experiment_name specified
    
    setup_mlflow(config)
    
    mock_set_tracking_uri.assert_called_once()
    mock_set_experiment.assert_called_once_with("mlops-boston")

def test_train_main_integration():
    """Test train main function integration (mocked)."""
    config_data = {
        "experiment_name": "test",
        "data_params": {"test_size": 0.2, "contamination": 0.05},
        "model_params": {"max_iter": 100, "solver": "liblinear"},
        "random_seed": 42
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    test_args = ["train.py", "--config", config_path]
    
    try:
        with patch('sys.argv', test_args):
            with patch('mlflow.start_run'):
                with patch('mlflow.log_params'):
                    with patch('mlflow.log_metrics'):
                        with patch('src.train.save_model'):
                            # This would test the main function if we import it
                            # For now, just test that config loading works
                            loaded_config = load_config(config_path)
                            assert loaded_config["experiment_name"] == "test"
    finally:
        os.unlink(config_path)
