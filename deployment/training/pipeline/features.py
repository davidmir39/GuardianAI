"""
features.py -- Ingenieria de caracteristicas y construccion del preprocesador.

Encapsula la definicion del ColumnTransformer (StandardScaler para continuas,
OneHotEncoder para categoricas, passthrough para binarias). Este artefacto
se serializa con joblib para que el contenedor de inferencia lo cargue tal
cual y aplique exactamente las mismas transformaciones.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import COLUMNAS_CATEGORICAS, COLUMNAS_CONTINUAS, TARGET_COL

logger = logging.getLogger(__name__)


def construir_preprocesador() -> ColumnTransformer:
    """Construye el ColumnTransformer del pipeline.

    - StandardScaler en columnas continuas.
    - OneHotEncoder (sparse_output=False, handle_unknown='ignore') en categoricas.
    - Resto de columnas (binarias/enteras) en passthrough.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), COLUMNAS_CONTINUAS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUMNAS_CATEGORICAS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en features y target."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Columna objetivo '{TARGET_COL}' ausente.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info("Separadas %s columnas de features y la columna objetivo.", X.shape[1])
    return X, y
