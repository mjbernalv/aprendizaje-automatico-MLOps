from __future__ import annotations
from src.mlops_example.data import load_boston_as_regression, load_boston_as_classification, train_test_split_xy
from src.mlops_example.io_utils import timestamped_dir, ensure_latest_symlink, save_model
from src.mlops_example.metrics import regression_metrics, classification_metrics
from src.mlops_example.modeling import build_model
from typing import Dict, Any
import mlflow.sklearn
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
    parser = argparse.ArgumentParser(description = "Entrenamiento con MLflow + Joblib")
    parser.add_argument("--config", type = str, default = "configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_mlflow(config)

    seed = int(config.get("seed", 42))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    model_type = config.get("model", {}).get("type", "rf")
    model_params = config.get("model", {}).get("params", {})
    dataset_name = config.get("dataset", {}).get("name", "boston")

    if model_type == "logreg":
        X, y = load_boston_as_classification(seed)
        task = "classification"
    else:
        X, y = load_boston_as_regression(seed)
        task = "regression"

    X_train, X_test, y_train, y_test = train_test_split_xy(X, y, test_size = test_size, seed = seed)

    with mlflow.start_run(run_name = f"{model_type}-{task}"):
        mlflow.log_params({"model_type": model_type,
                           "task": task,
                           "test_size": test_size,
                           **{f"param_{k}": v for k, v in model_params.items()}})

        model = build_model(model_type, **model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if task == "regression":
            mets = regression_metrics(y_test, y_pred)
        else:
            mets = classification_metrics(y_test, y_pred)

        mlflow.log_metrics(mets)

        out_dir = timestamped_dir(config.get("outputs", {}).get("dir", "artifacts"))
        model_path = os.path.join(out_dir, "model.joblib")
        save_model(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        ensure_latest_symlink(config.get("outputs", {}).get("dir", "artifacts"))

        print("\n=== Resultados ===")
        for k, v in mets.items():
            print(f"{k}: {v}")
            print(f"Modelo guardado en: {model_path}")
            print("Rastrea el experimento con: make mlflow-ui (http://localhost:5000)")


if __name__ == "__main__":
    main()