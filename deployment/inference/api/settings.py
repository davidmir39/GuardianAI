"""
settings.py -- Configuracion del contenedor de inferencia.

Todas las rutas y umbrales son configurables por variables de entorno
(regla del Hito 4: "rutas siempre configurables").
"""

from __future__ import annotations

import os
from pathlib import Path

# Carpeta con los artefactos publicados por el contenedor de entrenamiento.
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))

# Nombres canonicos de los artefactos.
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "modelo_final.joblib")
PREPROCESSOR_FILENAME = os.getenv("PREPROCESSOR_FILENAME", "preprocesador.joblib")
METADATA_FILENAME = os.getenv("METADATA_FILENAME", "metadatos.json")

# Umbral por defecto si los metadatos no lo incluyen.
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Permite forzar manualmente un umbral (sobreescribe el de los metadatos).
THRESHOLD_OVERRIDE = os.getenv("THRESHOLD_OVERRIDE")

# Configuracion de la API.
API_TITLE = os.getenv("API_TITLE", "GuardianAI - Inference API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_DESCRIPTION = os.getenv(
    "API_DESCRIPTION",
    "Servicio MLOps para deteccion de fraude en tiempo real "
    "(Bank Account Fraud Suite - NeurIPS 2022).",
)
