# syntax=docker/dockerfile:1.6
# ----------------------------------------------------------------------
# Contenedor de ENTRENAMIENTO de GuardianAI (Hito 4).
# Genera modelo_final.joblib, preprocesador.joblib y metadatos.json en
# la carpeta /artifacts (volumen compartido con el contenedor de inferencia).
# ----------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATASET_PATH=/data/raw/Base.csv \
    ARTIFACTS_DIR=/artifacts \
    REPORTS_DIR=/artifacts/reports

# libgomp1 es necesario para xgboost (OpenMP runtime).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalacion de dependencias antes del codigo para aprovechar la cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Codigo del pipeline.
COPY pipeline ./pipeline

# Crea las carpetas con permisos correctos por si los volumenes vienen vacios.
RUN mkdir -p /artifacts /data/raw

# Ejecucion estandar: corre el pipeline completo.
ENTRYPOINT ["python", "-m", "pipeline.train"]
CMD []
