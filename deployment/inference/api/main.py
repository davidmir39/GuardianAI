"""
main.py -- Aplicacion FastAPI de inferencia (Hito 4).

Carga modelo y preprocesador una sola vez al arrancar (lifespan) y expone:
- GET  /            redireccion conveniente a /docs.
- GET  /health      diagnostico del servicio.
- GET  /metadata    metadatos del modelo activo (metricas, fechas, columnas).
- POST /predict     prediccion para una unica transaccion.
- POST /predict/batch  prediccion para un lote de hasta 10.000 transacciones.

Errores:
- 422  cuerpo invalido (validacion Pydantic).
- 500  fallo interno (modelo no cargado, etc.).
- 503  servicio no listo (artefactos ausentes al arrancar).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


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
    try:
        salidas = pred.predict([transaccion.model_dump()])
    except Exception as exc:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=400, detail=f"Error en la prediccion: {exc}")
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
    return PrediccionLoteResponse(
        model_version=app.state.predictor.metadatos.get("modelo", "n/a"),
        n=len(salidas),
        predicciones=salidas,
    )
