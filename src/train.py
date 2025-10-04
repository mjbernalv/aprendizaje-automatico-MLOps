from __future__ import annotations
from src.data import load_data, train_test_split_xy, normalize_data, remove_outliers
from src.io_utils import timestamped_dir, ensure_latest_symlink, save_model
from src.metrics import classification_metrics
from src.modeling import ModelBuilder
from typing import Dict, Any
import mlflow
import argparse
import mlflow
import yaml
import os


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    :param path: Path to the YAML configuration file
    :return: Dictionary with the configuration parameters
    """
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(config: Dict[str, Any]):
    """"
    Sets up MLflow tracking URI and experiment.

    :param config: Configuration dictionary containing MLflow settings
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("experiment_name", "mlops-boston")
    mlflow.set_experiment(experiment_name)


def main():
    """
    Main function to execute the training pipeline.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description = "Entrenamiento del modelo")
    parser.add_argument("--config", type = str, default = "configs/train_config.yaml")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_mlflow(config)

    # Experiment parameters
    seed = int(config.get("seed", 42))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    model_name = config.get("model", {}).get("name", "RandomForestClassifier")
    model_params = config.get("model", {}).get("params", {})

    # Data loading and preprocessing
    X, y = load_data()
    X = normalize_data(X)
    X, y = remove_outliers(X, y, contamination = 0.05)
    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size = test_size, seed = seed)

    with mlflow.start_run(run_name = f"{model_name}"):
        mlflow.log_params({"model_name": model_name,
                           "test_size": test_size,
                           **{f"param_{k}": v for k, v in model_params.items()}})

        # Model training
        model_builder = ModelBuilder(random_state = seed, params = model_params)
        model_builder.train_model(X_train, y_train)

        # Model evaluation
        y_pred = model_builder.predict(X_test)

        # Calculate and log metrics
        metrics = classification_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # Save the model
        out_dir = timestamped_dir(config.get("outputs", {}).get("dir", "artifacts"))
        model_path = os.path.join(out_dir, "model.joblib")
        save_model(model_builder.model, model_path)
        mlflow.log_artifact(model_path, artifact_path = "model")

        ensure_latest_symlink(config.get("outputs", {}).get("dir", "artifacts"))

        print("\n------ Resultados ------")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            print(f"Modelo guardado en: {model_path}")
            print("Rastrea el experimento con: make mlflow-ui (http://localhost:5000)")


if __name__ == "__main__":
    main()