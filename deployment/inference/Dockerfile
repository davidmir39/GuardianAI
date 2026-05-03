# syntax=docker/dockerfile:1.6
# ----------------------------------------------------------------------
# Contenedor de INFERENCIA de GuardianAI (Hito 4).
# Carga modelo + preprocesador desde /artifacts y expone una API REST
# en el puerto 8000 (FastAPI + uvicorn).
# ----------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARTIFACTS_DIR=/artifacts \
    UVICORN_WORKERS=1

# libgomp1: runtime OpenMP requerido por xgboost.
# curl: para healthcheck nativo de docker compose.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api ./api

EXPOSE 8000

# Healthcheck simple contra el endpoint /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"ok"' || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
