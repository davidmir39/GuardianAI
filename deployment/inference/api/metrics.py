"""
metrics.py -- Cálculo de métricas del modelo con etiquetas diferidas (Hito 5).

Patrón de uso:
  1. La API genera un request_id por cada predicción y lo registra en el tracker
     junto con la probabilidad y la etiqueta predicha.
  2. Cuando llega /feedback con la etiqueta real (ground truth diferido), se
     correlaciona por request_id y se calcula la métrica sobre la ventana móvil
     de las últimas N predicciones confirmadas.
  3. El snapshot se publica vía /model-metrics y las gauges de Prometheus.

Limitación conocida (documentada en la memoria): en producción real el ground
truth llega con latencia de días o semanas (reclamaciones, chargebacks). En el
demo se simula vía simulate_anomaly.py para poder mostrar el flujo end-to-end.
"""

from __future__ import annotations

import os
from datetime import datetime
from threading import Lock
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

VENTANA_METRICAS = int(os.getenv("METRICS_WINDOW", "200"))
CACHE_PREDICCIONES_PENDIENTES = int(os.getenv("METRICS_PENDING_CACHE", "5000"))


class ModelMetricsTracker:
    """Mantiene una ventana móvil con las últimas N predicciones confirmadas
    (predicción + etiqueta real) y publica métricas agregadas.

    Es thread-safe: la API puede recibir /predict y /feedback en paralelo.
    """

    def __init__(
        self,
        ventana: int = VENTANA_METRICAS,
        cache_predicciones: int = CACHE_PREDICCIONES_PENDIENTES,
    ) -> None:
        self.ventana = ventana
        self._cache_max = cache_predicciones
        self._lock = Lock()
        self._pendientes: Dict[str, Dict] = {}
        self._confirmados: list = []
        self._ultima_actualizacion: Optional[datetime] = None
        self._cache_snapshot: Dict = self._snapshot_vacio()

    # ------------------------------------------------------------------ #
    # API pública                                                        #
    # ------------------------------------------------------------------ #
    def registrar_prediccion(
        self, request_id: str, probabilidad: float, etiqueta_pred: int
    ) -> None:
        """Guarda una predicción a la espera de su etiqueta real."""
        with self._lock:
            self._pendientes[request_id] = {
                "prob": float(probabilidad),
                "etiqueta_pred": int(etiqueta_pred),
                "ts": datetime.utcnow(),
            }
            self._purgar_pendientes_si_excede()

    def registrar_feedback(self, request_id: str, etiqueta_real: int) -> bool:
        """Asocia un ground truth con su predicción. Devuelve True si se pudo
        correlacionar; False si la predicción ya no estaba en caché."""
        with self._lock:
            pred = self._pendientes.pop(request_id, None)
            if pred is None:
                return False
            self._confirmados.append(
                {
                    "prob": pred["prob"],
                    "etiqueta_pred": pred["etiqueta_pred"],
                    "etiqueta_real": int(etiqueta_real),
                }
            )
            if len(self._confirmados) > self.ventana:
                self._confirmados = self._confirmados[-self.ventana :]
            self._ultima_actualizacion = datetime.utcnow()
            self._recalcular()
            return True

    def snapshot(self) -> Dict:
        """Devuelve el último snapshot calculado (copia)."""
        with self._lock:
            return dict(self._cache_snapshot)

    # ------------------------------------------------------------------ #
    # Internos                                                           #
    # ------------------------------------------------------------------ #
    def _snapshot_vacio(self) -> Dict:
        return {
            "n_confirmados": 0,
            "ventana": self.ventana,
            "f1": None,
            "auc": None,
            "precision": None,
            "recall": None,
            "ultima_actualizacion": None,
        }

    def _purgar_pendientes_si_excede(self) -> None:
        """Evita que el dict de pendientes crezca sin límite."""
        if len(self._pendientes) <= self._cache_max:
            return
        ordenados = sorted(self._pendientes.items(), key=lambda kv: kv[1]["ts"])
        sobrantes = len(self._pendientes) - self._cache_max
        for clave, _ in ordenados[:sobrantes]:
            self._pendientes.pop(clave, None)

    def _recalcular(self) -> None:
        n = len(self._confirmados)
        if n < 2:
            self._cache_snapshot = {
                **self._snapshot_vacio(),
                "n_confirmados": n,
                "ultima_actualizacion": (
                    self._ultima_actualizacion.isoformat()
                    if self._ultima_actualizacion
                    else None
                ),
            }
            return

        y_true = np.array([r["etiqueta_real"] for r in self._confirmados])
        y_pred = np.array([r["etiqueta_pred"] for r in self._confirmados])
        y_prob = np.array([r["prob"] for r in self._confirmados])

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

        # AUC requiere al menos una muestra de cada clase.
        if len(np.unique(y_true)) == 2:
            auc: Optional[float] = float(roc_auc_score(y_true, y_prob))
        else:
            auc = None

        self._cache_snapshot = {
            "n_confirmados": n,
            "ventana": self.ventana,
            "f1": f1,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "ultima_actualizacion": self._ultima_actualizacion.isoformat()
            if self._ultima_actualizacion
            else None,
        }
