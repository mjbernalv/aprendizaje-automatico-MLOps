.PHONY: install train predict mlflow-ui test

install:
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m src.train --config configs/train_config.yaml

predict:
	. .venv/bin/activate && python -m src.predict \
	--model_path artifacts/latest/model.joblib \
	--samples_file artifacts/latest/sample_inputs.csv \
	--output_csv artifacts/latest/predictions.csv

mlflow-ui:
	. .venv/bin/activate && mlflow ui --backend-store-uri file:./mlruns --port 5000

test:
	. .venv/bin/activate && python -m pytest -q