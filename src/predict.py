from __future__ import annotations
from src.io_utils import load_model
import pandas as pd
import argparse
import os


def main():
    """
    Main function to perform inference using a trained model (joblib).
    """
    parser = argparse.ArgumentParser(description = "Inferencia usando modelo entrenado (joblib)")
    parser.add_argument("--model_path", type = str, default = "artifacts/latest/model.joblib")
    parser.add_argument("--samples_file", type = str, default = "")
    parser.add_argument("--output_csv", type = str, default = "artifacts/latest/predictions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No existe el modelo en {args.model_path}. Corre 'make train' primero.")

    model = load_model(args.model_path)

    if args.samples_file and os.path.exists(args.samples_file):
        X = pd.read_csv(args.samples_file)
    else:
        X = pd.DataFrame({"CRIM": [0.03, 0.1],
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

    preds = model.predict(X)
    print("Predicciones:", preds)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok = True)
    pd.DataFrame({"prediction": preds}).to_csv(args.output_csv, index = False)
    print(f"Predicciones guardadas en: {args.output_csv}")


if __name__ == "__main__":
    main()