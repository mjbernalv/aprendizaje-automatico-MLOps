# aprendizaje-automatico-MLOps
Repositorio para la exposición de MLOps de la materia Aprendizaje Automático de la Maestría en Ciencias de los Datos y Analítica de EAFIT

Este repo muestra un **pipeline simplificado y reproducible**:
Este repositorio muestra un ejemplo sencillo de MLops y de la integración de MLflow a la experimentación en proyectos de ML.

## Generalidades
- **Dataset**: Boston Housing.
- **Modelo**: `LogisticRegression` para clasificación binaria.
- **Experimentos**: Jupyter notebook para realizar experimentos.
- **Guardado del modelo**: `joblib` → `artifacts/<timestamp>/model.joblib` y enlace `artifacts/latest/`.
- **Tracking**: MLflow (parámetros, métricas, artefactos). UI local con `make mlflow-ui`.
- **Inferencia**: `src/predict.py` carga el modelo y genera `predictions.csv`.
- **Pruebas**: `pytest` básico.

## Quickstart
```bash
# 1) Crear entorno e instalar dependencias
make install

# 2) Entrenar (usa configs/train_config.yaml)
make train

# 3) Ver experimentos en MLflow UI
make mlflow-ui # http://localhost:5000

# 4) Inferir con el último modelo
make predict