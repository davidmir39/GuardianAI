"""
main.py -- Aplicacion FastAPI de inferencia (Hito 5).

Cambios Hito 5:
- Instrumentacion Prometheus (metricas HTTP + CPU/RAM custom).
- Logging de predicciones a /logs/predictions_log.jsonl para monitorización.
- request_id por prediccion para correlacionar con feedback diferido.
- Endpoint /feedback: recibe el ground truth real con latencia simulada.
- Endpoint /model-metrics: snapshot de F1/AUC/precision/recall sobre la
  ventana movil de predicciones confirmadas.
- Endpoint /reload: recarga el modelo en caliente tras un reentrenamiento.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from .metrics import ModelMetricsTracker
from .predictor import FraudPredictor
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LoteTransacciones,
    MetadataResponse,
    ModelMetricsResponse,
    PrediccionItem,
    PrediccionLoteItem,
    PrediccionLoteResponse,
    PrediccionResponse,
    ReloadResponse,
    Transaccion,
)
from .settings import API_DESCRIPTION, API_TITLE, API_VERSION, ARTIFACTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------- #
# Configuracion de logs (Hito 5)                                         #
# ---------------------------------------------------------------------- #
LOGS_DIR = Path(os.getenv("LOGS_DIR", "/logs"))
PREDICTIONS_LOG = LOGS_DIR / "predictions_log.jsonl"
FEEDBACK_LOG = LOGS_DIR / "feedback_log.jsonl"

# ---------------------------------------------------------------------- #
# Metricas Prometheus custom (Hito 5)                                    #
# ---------------------------------------------------------------------- #
CPU_GAUGE = Gauge("guardianai_cpu_percent", "Uso de CPU del sistema (%)")
RAM_GAUGE = Gauge("guardianai_ram_percent", "Uso de RAM del sistema (%)")
FRAUD_RATE_GAUGE = Gauge(
    "guardianai_fraud_rate", "Tasa de fraude en las ultimas predicciones"
)
PREDICTIONS_TOTAL_GAUGE = Gauge(
    "guardianai_predictions_total", "Total de predicciones registradas"
)
MODEL_F1_GAUGE = Gauge("guardianai_model_f1", "F1 score sobre ventana confirmada")
MODEL_AUC_GAUGE = Gauge("guardianai_model_auc", "ROC-AUC sobre ventana confirmada")
MODEL_PRECISION_GAUGE = Gauge(
    "guardianai_model_precision", "Precision sobre ventana confirmada"
)
MODEL_RECALL_GAUGE = Gauge(
    "guardianai_model_recall", "Recall sobre ventana confirmada"
)
MODEL_CONFIRMED_GAUGE = Gauge(
    "guardianai_model_confirmados", "Predicciones confirmadas en la ventana"
)


def _actualizar_metricas_sistema() -> None:
    """Actualiza los gauges de CPU y RAM con los valores actuales."""
    CPU_GAUGE.set(psutil.cpu_percent(interval=None))
    RAM_GAUGE.set(psutil.virtual_memory().percent)


def _actualizar_gauges_modelo(snapshot: dict) -> None:
    """Publica el snapshot de métricas del modelo en Prometheus."""
    MODEL_CONFIRMED_GAUGE.set(snapshot.get("n_confirmados", 0))
    for clave, gauge in (
        ("f1", MODEL_F1_GAUGE),
        ("auc", MODEL_AUC_GAUGE),
        ("precision", MODEL_PRECISION_GAUGE),
        ("recall", MODEL_RECALL_GAUGE),
    ):
        valor = snapshot.get(clave)
        if valor is not None:
            gauge.set(valor)


def _log_prediccion(request_id: str, features: dict, salida: dict) -> None:
    """Escribe una prediccion en el log JSONL para monitorización."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        registro = {
            "request_id": request_id,
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


def _log_feedback(request_id: str, fraud_bool_real: int, registrado: bool) -> None:
    """Persiste el feedback para auditoria y futuros reentrenamientos."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        registro = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "fraud_bool_real": int(fraud_bool_real),
            "registrado": bool(registrado),
        }
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro) + "\n")
    except Exception as exc:
        logger.warning("No se pudo escribir en el log de feedback: %s", exc)


# ---------------------------------------------------------------------- #
# Ciclo de vida: carga del predictor al arrancar el servidor.            #
# ---------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servicio. Cargando artefactos desde %s ...", ARTIFACTS_DIR)
    app.state.metrics_tracker = ModelMetricsTracker()
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
    salida = salidas[0]
    request_id = uuid.uuid4().hex
    _log_prediccion(request_id, features, salida)
    app.state.metrics_tracker.registrar_prediccion(
        request_id, salida["probabilidad_fraude"], salida["etiqueta"]
    )
    _actualizar_metricas_sistema()
    return PrediccionResponse(
        model_version=app.state.predictor.metadatos.get("modelo", "n/a"),
        request_id=request_id,
        prediccion=PrediccionItem(**salida),
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

    items: list[PrediccionLoteItem] = []
    for features, salida in zip(registros, salidas):
        request_id = uuid.uuid4().hex
        _log_prediccion(request_id, features, salida)
        app.state.metrics_tracker.registrar_prediccion(
            request_id, salida["probabilidad_fraude"], salida["etiqueta"]
        )
        items.append(
            PrediccionLoteItem(
                request_id=request_id,
                prediccion=PrediccionItem(**salida),
            )
        )

    if salidas:
        tasa = sum(s["etiqueta"] for s in salidas) / len(salidas)
        FRAUD_RATE_GAUGE.set(tasa)
    PREDICTIONS_TOTAL_GAUGE.set(
        PREDICTIONS_LOG.stat().st_size if PREDICTIONS_LOG.exists() else 0
    )
    _actualizar_metricas_sistema()
    return PrediccionLoteResponse(
        model_version=app.state.predictor.metadatos.get("modelo", "n/a"),
        n=len(items),
        predicciones=items,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(payload: FeedbackRequest):
    """Recibe el ground truth diferido de una predicción previa.

    En producción real este endpoint sería llamado por el sistema de back-office
    cuando se confirma una reclamación o chargeback. Aquí lo usa
    simulate_anomaly.py para alimentar las métricas F1/AUC/precisión.
    """
    tracker: ModelMetricsTracker = app.state.metrics_tracker
    registrado = tracker.registrar_feedback(payload.request_id, payload.fraud_bool_real)
    _log_feedback(payload.request_id, payload.fraud_bool_real, registrado)

    snapshot = tracker.snapshot()
    _actualizar_gauges_modelo(snapshot)

    if not registrado:
        return FeedbackResponse(
            status="ignored",
            request_id=payload.request_id,
            detalle="request_id no encontrado en cache (puede haber expirado).",
            n_confirmados=snapshot.get("n_confirmados", 0),
        )
    return FeedbackResponse(
        status="ok",
        request_id=payload.request_id,
        n_confirmados=snapshot.get("n_confirmados", 0),
    )


@app.get(
    "/model-metrics",
    response_model=ModelMetricsResponse,
    tags=["feedback"],
)
def model_metrics():
    """Snapshot de las métricas del modelo sobre la ventana móvil confirmada."""
    snapshot = app.state.metrics_tracker.snapshot()
    return ModelMetricsResponse(**snapshot)


@app.post("/reload", response_model=ReloadResponse, tags=["meta"])
def reload_model():
    """Recarga el modelo, preprocesador y metadatos desde disco.

    Pensado para cerrar el feedback loop: tras un reentrenamiento exitoso, el
    monitor invoca este endpoint para que el modelo nuevo entre en producción
    sin reiniciar el contenedor.
    """
    try:
        nuevo = FraudPredictor()
    except Exception as exc:
        logger.exception("Error recargando el modelo")
        raise HTTPException(
            status_code=500, detail=f"Error recargando el modelo: {exc}"
        )

    app.state.predictor = nuevo
    app.state.error = None
    # Reseteo del tracker: el modelo nuevo no debería heredar métricas del anterior.
    app.state.metrics_tracker = ModelMetricsTracker()
    _actualizar_gauges_modelo(app.state.metrics_tracker.snapshot())

    info = nuevo.info()
    logger.info("Modelo recargado: %s (umbral=%.4f)", info["modelo"], nuevo.threshold)
    return ReloadResponse(
        status="ok",
        modelo=info["modelo"],
        umbral_activo=nuevo.threshold,
        fecha_entrenamiento=info.get("fecha_entrenamiento"),
    )
