"""
main.py -- Aplicacion FastAPI de inferencia (Hito 5).

Cambios Hito 5:
- Instrumentacion Prometheus (metricas HTTP + CPU/RAM custom).
- Logging de predicciones a /logs/predictions_log.jsonl para monitorización.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

from .predictor import FraudPredictor
from .schemas import (
    HealthResponse,
    LoteTransacciones,
    MetadataResponse,
    PrediccionLoteResponse,
    PrediccionResponse,
    Transaccion,
)
from .settings import API_DESCRIPTION, API_TITLE, API_VERSION, ARTIFACTS_DIR

import json
import os
from datetime import datetime
from pathlib import Path

import psutil
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------- #
# Configuracion de logs de prediccion (Hito 5)                          #
# ---------------------------------------------------------------------- #
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/logs"))
PREDICTIONS_LOG = LOGS_DIR / "predictions_log.jsonl"

# ---------------------------------------------------------------------- #
# Metricas Prometheus custom (Hito 5)                                   #
# ---------------------------------------------------------------------- #
CPU_GAUGE = Gauge("guardianai_cpu_percent", "Uso de CPU del sistema (%)")
RAM_GAUGE = Gauge("guardianai_ram_percent", "Uso de RAM del sistema (%)")
FRAUD_RATE_GAUGE = Gauge("guardianai_fraud_rate", "Tasa de fraude en las ultimas predicciones")
PREDICTIONS_TOTAL_GAUGE = Gauge("guardianai_predictions_total", "Total de predicciones registradas")


def _actualizar_metricas_sistema() -> None:
    """Actualiza los gauges de CPU y RAM con los valores actuales."""
    CPU_GAUGE.set(psutil.cpu_percent(interval=None))
    RAM_GAUGE.set(psutil.virtual_memory().percent)


def _log_prediccion(features: dict, salida: dict) -> None:
    """Escribe una prediccion en el log JSONL para monitorización."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        registro = {
            "timestamp": datetime.utcnow().isoformat(),
            **features,
            "probabilidad_fraude": salida["probabilidad_fraude"],
            "etiqueta": salida["etiqueta"],
            "decision": salida["decision"],
        }
        with open(PREDICTIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro) + "\n")
    except Exception as exc:
        logger.warning("No se pudo escribir en el log de predicciones: %s", exc)


# ---------------------------------------------------------------------- #
# Ciclo de vida: carga del predictor al arrancar el servidor.            #
# ---------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servicio. Cargando artefactos desde %s ...", ARTIFACTS_DIR)
    try:
        app.state.predictor = FraudPredictor()
        logger.info("Servicio listo. Info: %s", app.state.predictor.info())
    except FileNotFoundError as exc:
        logger.error("No se pudo arrancar el predictor: %s", exc)
        app.state.predictor = None
        app.state.error = str(exc)
    yield
    logger.info("Apagando servicio.")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# Instrumentacion Prometheus
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app)

def _get_predictor(app: FastAPI) -> FraudPredictor:
    pred = getattr(app.state, "predictor", None)
    if pred is None or not pred.listo:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(app.state, "error", "Predictor no inicializado."),
        )
    return pred


# ---------------------------------------------------------------------- #
# Endpoints                                                              #
# ---------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    _actualizar_metricas_sistema()
    pred = getattr(app.state, "predictor", None)
    if pred is None or not pred.listo:
        return HealthResponse(
            status="degraded",
            artifacts_dir=str(ARTIFACTS_DIR),
            modelo_cargado=False,
            umbral_activo=None,
        )
    return HealthResponse(
        status="ok",
        artifacts_dir=str(ARTIFACTS_DIR),
        modelo_cargado=True,
        umbral_activo=pred.threshold,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["meta"])
def metadata():
    pred = _get_predictor(app)
    info = pred.info()
    return MetadataResponse(
        modelo=info["modelo"],
        umbral_activo=info["umbral_activo"],
        fecha_entrenamiento=info.get("fecha_entrenamiento"),
        metricas_test=info.get("metricas_test", {}),
        n_columnas_continuas=info.get("n_columnas_continuas", 0),
        n_columnas_categoricas=info.get("n_columnas_categoricas", 0),
    )


@app.post("/predict", response_model=PrediccionResponse, tags=["inference"])
def predict(transaccion: Transaccion):
    pred = _get_predictor(app)
    features = transaccion.model_dump()
    try:
        salidas = pred.predict([features])
    except Exception as exc:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=400, detail=f"Error en la prediccion: {exc}")
    _log_prediccion(features, salidas[0])
    _actualizar_metricas_sistema()
    return PrediccionResponse(
        model_version=app.state.predictor.metadatos.get("modelo", "n/a"),
        prediccion=salidas[0],
    )


@app.post(
    "/predict/batch",
    response_model=PrediccionLoteResponse,
    tags=["inference"],
)
def predict_batch(lote: LoteTransacciones):
    pred = _get_predictor(app)
    try:
        registros = [t.model_dump() for t in lote.transacciones]
        salidas = pred.predict(registros)
    except Exception as exc:
        logger.exception("Error en /predict/batch")
        raise HTTPException(status_code=400, detail=f"Error en la prediccion: {exc}")
    for features, salida in zip(registros, salidas):
        _log_prediccion(features, salida)
    if salidas:
        tasa = sum(s["etiqueta"] for s in salidas) / len(salidas)
        FRAUD_RATE_GAUGE.set(tasa)
    PREDICTIONS_TOTAL_GAUGE.set(PREDICTIONS_LOG.stat().st_size if PREDICTIONS_LOG.exists() else 0)
    _actualizar_metricas_sistema()
    return PrediccionLoteResponse(
        model_version=app.state.predictor.metadatos.get("modelo", "n/a"),
        n=len(salidas),
        predicciones=salidas,
    )
