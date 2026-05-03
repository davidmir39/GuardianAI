"""
predictor.py -- Wrapper de carga y prediccion del modelo.

Aisla la dependencia con el modelo: carga modelo, preprocesador y metadatos
una sola vez al arrancar el contenedor y expone un metodo `predict` que
devuelve probabilidad y etiqueta dada la transaccion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from .settings import (
    ARTIFACTS_DIR,
    DEFAULT_THRESHOLD,
    METADATA_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    THRESHOLD_OVERRIDE,
)

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Wrapper alrededor del modelo + preprocesador entrenado."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.modelo = None
        self.preprocesador = None
        self.metadatos: Dict[str, Any] = {}
        self.threshold: float = DEFAULT_THRESHOLD
        self._cargar()

    # -------------------------------------------------------------- #
    # Carga de artefactos                                            #
    # -------------------------------------------------------------- #
    def _cargar(self) -> None:
        ruta_modelo = self.artifacts_dir / MODEL_FILENAME
        ruta_pre = self.artifacts_dir / PREPROCESSOR_FILENAME
        ruta_meta = self.artifacts_dir / METADATA_FILENAME

        if not ruta_modelo.exists() or not ruta_pre.exists():
            raise FileNotFoundError(
                f"Faltan artefactos en {self.artifacts_dir}. "
                "Ejecute primero el contenedor de entrenamiento "
                "('docker compose run --rm training') o copie los artefactos "
                "preentrenados desde models/."
            )

        logger.info("Cargando modelo desde %s", ruta_modelo)
        self.modelo = joblib.load(ruta_modelo)
        logger.info("Cargando preprocesador desde %s", ruta_pre)
        self.preprocesador = joblib.load(ruta_pre)

        if ruta_meta.exists():
            with open(ruta_meta, "r", encoding="utf-8") as fh:
                self.metadatos = json.load(fh)
            umbral_meta = (
                self.metadatos.get("metricas_test", {}).get("umbral_optimo")
            )
            if umbral_meta is not None:
                self.threshold = float(umbral_meta)

        if THRESHOLD_OVERRIDE is not None:
            try:
                self.threshold = float(THRESHOLD_OVERRIDE)
                logger.info(
                    "Usando umbral forzado por entorno: %.4f", self.threshold
                )
            except ValueError:
                logger.warning(
                    "THRESHOLD_OVERRIDE invalido (%s); se ignora.",
                    THRESHOLD_OVERRIDE,
                )

        logger.info("Predictor listo. Umbral activo: %.4f", self.threshold)

    # -------------------------------------------------------------- #
    # Inferencia                                                     #
    # -------------------------------------------------------------- #
    def _to_dataframe(self, registros: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(list(registros))
        # Coherencia con el preprocesador (mismas columnas que en entrenamiento).
        cols_categoricas = list(self.preprocesador.transformers_[1][2])
        for c in cols_categoricas:
            if c in df.columns:
                df[c] = df[c].astype("category")
        return df

    def predict(self, registros: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predice probabilidad y etiqueta para una lista de transacciones.

        Args:
            registros: iterable de diccionarios con las features de cada transaccion.

        Returns:
            Lista de dicts con probabilidad, etiqueta, umbral, decision y nivel de riesgo.
        """
        df = self._to_dataframe(registros)
        X = self.preprocesador.transform(df)
        probs = self.modelo.predict_proba(X)[:, 1]
        umbral = self.threshold

        salidas = []
        for prob in probs:
            prob_f = float(prob)
            etiqueta = int(prob_f >= umbral)
            salidas.append({
                "probabilidad_fraude": round(prob_f, 6),
                "etiqueta": etiqueta,
                "umbral": round(umbral, 6),
                "decision": "BLOQUEAR" if etiqueta == 1 else "PERMITIR",
                "nivel_riesgo": self._nivel_riesgo(prob_f, umbral),
            })
        return salidas

    @staticmethod
    def _nivel_riesgo(prob: float, umbral: float) -> str:
        """Categoriza el riesgo en BAJO / MEDIO / ALTO / CRITICO.

        El criterio es relativo al umbral activo: por debajo del 50% del umbral
        es BAJO; entre 50% y umbral es MEDIO; sobre umbral es ALTO; sobre
        umbral + 5 puntos porcentuales es CRITICO. Esto facilita su lectura
        para analistas sin requerir conocer el umbral exacto.
        """
        if prob < umbral * 0.5:
            return "BAJO"
        if prob < umbral:
            return "MEDIO"
        if prob < umbral + 0.05:
            return "ALTO"
        return "CRITICO"

    # -------------------------------------------------------------- #
    # Introspeccion                                                  #
    # -------------------------------------------------------------- #
    def info(self) -> Dict[str, Any]:
        """Devuelve un resumen del modelo activo."""
        return {
            "modelo": self.metadatos.get("modelo", "desconocido"),
            "umbral_activo": self.threshold,
            "fecha_entrenamiento": self.metadatos.get("fecha_entrenamiento"),
            "metricas_test": self.metadatos.get("metricas_test", {}),
            "n_columnas_continuas": len(self.metadatos.get("columnas_continuas", [])),
            "n_columnas_categoricas": len(self.metadatos.get("columnas_categoricas", [])),
        }

    @property
    def listo(self) -> bool:
        return self.modelo is not None and self.preprocesador is not None
