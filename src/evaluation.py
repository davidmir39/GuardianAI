"""
evaluation.py — Funciones comunes de evaluacion para los notebooks del Hito 3.

Centraliza el calculo de metricas y la generacion de graficos reutilizables
para mantener coherencia entre NB02, NB03, NB04 y NB05.

Todas las metricas se evaluan sobre el conjunto de **test original
desbalanceado** (prevalencia de fraude ~1.1 %), que representa las
condiciones reales de operacion.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------------ #
# Metricas                                                            #
# ------------------------------------------------------------------ #
def buscar_umbral_optimo(y_true, y_prob) -> float:
    """Busca el umbral que maximiza F1 sobre una curva Precision-Recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Evitar division por cero
    f1_scores = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-12)
    # precision_recall_curve devuelve len(thresholds) = len(precisions) - 1
    idx_opt = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[idx_opt])


def evaluar_modelo(
    model,
    X_test,
    y_test,
    nombre: str = "Modelo",
    tiempo_entrenamiento: Optional[float] = None,
    imprimir: bool = True,
) -> Dict[str, Any]:
    """Evalua un modelo sobre el test real desbalanceado.

    Devuelve un dict con las metricas principales. Si `imprimir=True`,
    tambien muestra un resumen por stdout.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    umbral = buscar_umbral_optimo(y_test, y_prob)
    y_pred = (y_prob >= umbral).astype(int)

    metricas = {
        "nombre": nombre,
        "umbral_optimo": umbral,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "tiempo_entrenamiento": tiempo_entrenamiento,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    if imprimir:
        print(f"{nombre} -- evaluado sobre test real (desbalanceado)")
        if tiempo_entrenamiento is not None:
            print(f"  Tiempo de entrenamiento: {tiempo_entrenamiento:.1f}s")
        print(f"  Umbral optimo (F1 max):  {umbral:.2f}")
        print(f"  Precision:  {metricas['precision']:.4f}")
        print(f"  Recall:     {metricas['recall']:.4f}")
        print(f"  F1-Score:   {metricas['f1']:.4f}")
        print(f"  PR-AUC:     {metricas['pr_auc']:.4f}")
        print(f"  ROC-AUC:    {metricas['roc_auc']:.4f}")

    return metricas


def resumen_metricas_tabla(resultados: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Devuelve un DataFrame con la comparativa de metricas de varios modelos.

    `resultados` es un diccionario {nombre_modelo: dict_metricas}
    como el que devuelve `evaluar_modelo`.
    """
    filas = []
    for nombre, m in resultados.items():
        filas.append(
            {
                "Modelo": nombre,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
                "Umbral": round(m["umbral_optimo"], 2),
                "Tiempo (s)": (
                    round(m["tiempo_entrenamiento"], 1)
                    if m.get("tiempo_entrenamiento") is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ #
# Graficos                                                            #
# ------------------------------------------------------------------ #
def plot_curvas_roc_pr(
    resultados: Dict[str, Dict[str, Any]],
    y_test,
    titulo: str = "Curvas ROC y Precision-Recall",
    ruta_salida: Optional[str] = None,
) -> plt.Figure:
    """Genera en una sola figura la curva ROC y la curva Precision-Recall
    para varios modelos evaluados sobre el mismo `y_test`.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for nombre, m in resultados.items():
        y_prob = m["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC={m['roc_auc']:.3f})")

        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(recalls, precisions, label=f"{nombre} (AP={m['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Curva Precision-Recall")
    ax_pr.legend(loc="upper right", fontsize=8)
    ax_pr.grid(alpha=0.3)

    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    return fig


def plot_matriz_confusion(
    y_true,
    y_pred,
    nombre_modelo: str = "Modelo",
    normalizar: bool = True,
    ruta_salida: Optional[str] = None,
) -> plt.Figure:
    """Matriz de confusion normalizada por fila (o absoluta si normalizar=False)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalizar:
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalizar else cm_norm.max())

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm_norm[i, j]
            color = "white" if valor > (0.5 if normalizar else cm_norm.max() * 0.5) else "black"
            etiqueta = f"{valor:.2%}" if normalizar else f"{int(valor)}"
            ax.text(j, i, etiqueta, ha="center", va="center", color=color, fontsize=11)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No fraude", "Fraude"])
    ax.set_yticklabels(["No fraude", "Fraude"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Realidad")
    ax.set_title(
        f"Matriz de confusion — {nombre_modelo}"
        + (" (normalizada por fila)" if normalizar else "")
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if ruta_salida:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------ #
# Pipeline de preprocesamiento                                        #
# ------------------------------------------------------------------ #
def preparar_preprocesador(
    columnas_continuas: list,
    columnas_categoricas: list,
):
    """Construye un `ColumnTransformer` con StandardScaler + OneHotEncoder
    para las columnas indicadas. Mantiene coherencia entre notebooks.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_continuas),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                columnas_categoricas,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# ------------------------------------------------------------------ #
# Persistencia de modelos                                             #
# ------------------------------------------------------------------ #
def guardar_artefactos(
    directorio: str,
    modelo: Any,
    preprocesador: Optional[Any] = None,
    metadatos: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persiste modelo + preprocesador + metadatos en disco usando joblib.

    Los ficheros .joblib NO se suben al repo (ver .gitignore). Sirven para
    consumo local entre notebooks (por ejemplo NB05 entrena y NB04 carga).
    """
    import joblib

    os.makedirs(directorio, exist_ok=True)
    rutas: Dict[str, str] = {}

    ruta_modelo = os.path.join(directorio, "modelo_final.joblib")
    joblib.dump(modelo, ruta_modelo)
    rutas["modelo"] = ruta_modelo

    if preprocesador is not None:
        ruta_pre = os.path.join(directorio, "preprocesador.joblib")
        joblib.dump(preprocesador, ruta_pre)
        rutas["preprocesador"] = ruta_pre

    if metadatos is not None:
        import json

        ruta_meta = os.path.join(directorio, "metadatos.json")
        with open(ruta_meta, "w", encoding="utf-8") as fh:
            json.dump(metadatos, fh, indent=2, ensure_ascii=False)
        rutas["metadatos"] = ruta_meta

    return rutas


def cargar_artefactos(directorio: str) -> Dict[str, Any]:
    """Carga los artefactos que `guardar_artefactos` haya escrito previamente."""
    import joblib

    artefactos: Dict[str, Any] = {}
    ruta_modelo = os.path.join(directorio, "modelo_final.joblib")
    ruta_pre = os.path.join(directorio, "preprocesador.joblib")
    ruta_meta = os.path.join(directorio, "metadatos.json")

    if os.path.exists(ruta_modelo):
        artefactos["modelo"] = joblib.load(ruta_modelo)
    if os.path.exists(ruta_pre):
        artefactos["preprocesador"] = joblib.load(ruta_pre)
    if os.path.exists(ruta_meta):
        import json

        with open(ruta_meta, "r", encoding="utf-8") as fh:
            artefactos["metadatos"] = json.load(fh)

    return artefactos


# ------------------------------------------------------------------ #
# Cronometraje                                                        #
# ------------------------------------------------------------------ #
class Cronometro:
    """Cronometro sencillo con `with`-statement."""

    def __enter__(self) -> "Cronometro":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_args) -> None:
        self.elapsed = time.perf_counter() - self.t0

    @property
    def segundos(self) -> float:
        return getattr(self, "elapsed", time.perf_counter() - self.t0)
