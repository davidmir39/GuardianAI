"""
data_ingestion.py -- Carga del dataset crudo (Base.csv).

Aisla el unico punto de I/O hacia la fuente de datos. En produccion este
modulo se sustituiria por una conexion a un data warehouse o un bucket S3,
sin tocar el resto del pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATASET_PATH, DTYPE_BASE_CSV, TARGET_COL

logger = logging.getLogger(__name__)


def cargar_dataset(ruta: Optional[Path] = None) -> pd.DataFrame:
    """Carga Base.csv aplicando el mapa de dtypes para minimizar memoria.

    Args:
        ruta: Ruta al CSV. Si es None, usa la variable de entorno DATASET_PATH.

    Returns:
        DataFrame con todas las columnas del dataset.
    """
    ruta = ruta or DATASET_PATH
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en '{ruta}'. "
            "Monte el volumen con Base.csv o ajuste la variable DATASET_PATH."
        )

    logger.info("Cargando dataset desde %s", ruta)
    df = pd.read_csv(ruta, dtype=DTYPE_BASE_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"La columna objetivo '{TARGET_COL}' no esta presente en el CSV."
        )

    logger.info(
        "Dataset cargado: %s filas, %s columnas, prevalencia fraude = %.4f%%",
        len(df), df.shape[1], df[TARGET_COL].mean() * 100,
    )
    return df
