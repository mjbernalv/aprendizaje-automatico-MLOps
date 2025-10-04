.PHONY: install train predict mlflow-ui test

# Detect OS and set paths
ifeq ($(OS),Windows_NT)
    VENV_BIN := .venv/Scripts
    PYTHON := $(VENV_BIN)/python.exe
    PIP := $(VENV_BIN)/pip.exe
    MLFLOW := $(VENV_BIN)/mlflow.exe
else
    VENV_BIN := .venv/bin
    PYTHON := $(VENV_BIN)/python
    PIP := $(VENV_BIN)/pip
    MLFLOW := $(VENV_BIN)/mlflow
endif

install:
	python -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m src.train --config configs/train_config.yaml

predict:
	$(PYTHON) -m src.predict \
		--model_path artifacts/latest/model.joblib \
		--samples_file artifacts/latest/sample_inputs.csv \
		--output_csv artifacts/latest/predictions.csv

mlflow-ui:
	$(MLFLOW) ui --backend-store-uri file:./mlruns --port 5000

test:
	$(PYTHON) -m pytest -q
