.PHONY: install train predict mlflow-ui test

# Detect OS and set activation path
ifeq ($(OS),Windows_NT)
	ACTIVATE = .venv/Scripts/activate
	PYTHON = .venv/Scripts/python
	PIP = .venv/Scripts/pip
	MFLOW = .venv/Scripts/mlflow
else
	ACTIVATE = .venv/bin/activate
	PYTHON = .venv/bin/python
	PIP = .venv/bin/pip
	MFLOW = .venv/bin/mlflow
endif

install:
	python -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

train:
	. $(ACTIVATE) && $(PYTHON) -m src.train --config configs/train_config.yaml

predict:
	. $(ACTIVATE) && $(PYTHON) -m src.predict \
	--model_path artifacts/latest/model.joblib \
	--samples_file artifacts/latest/sample_inputs.csv \
	--output_csv artifacts/latest/predictions.csv

mlflow-ui:
	. $(ACTIVATE) && $(MFLOW) ui --backend-store-uri file:./mlruns --port 5000

test:
	. $(ACTIVATE) && $(PYTHON) -m pytest -q
