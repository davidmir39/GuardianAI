"""
train.py -- Orquestador del entrenamiento (Hito 4).

Reproduce, fuera del notebook, el flujo end-to-end del NB05:
1. Ingesta del CSV.
2. Split estratificado 70/15/15.
3. Construccion del preprocesador (StandardScaler + OneHotEncoder + passthrough).
4. Calculo de scale_pos_weight sobre train (sin leakage).
5. Busqueda de hiperparametros (modo 'quick' o 'full').
6. Refit en train + val con los mejores hiperparametros.
7. Busqueda de umbral optimo (maximiza F1) sobre test.
8. Persistencia de modelo + preprocesador + metadatos en ARTIFACTS_DIR.

Todas las rutas y opciones son configurables por variables de entorno o
flags de linea de comandos (regla del Hito 4: rutas siempre configurables).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from .config import (
    ARTIFACTS_DIR,
    BEST_PARAMS_HITO3,
    DATASET_PATH,
    N_JOBS,
    RANDOM_STATE,
    REPORTS_DIR,
    SEARCH_CV_FOLDS,
    SEARCH_MODE,
    SEARCH_N_ITER,
    SEARCH_SCORING,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from .data_ingestion import cargar_dataset
from .features import construir_preprocesador, separar_X_y

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training")


# ------------------------------------------------------------------ #
# Utilidades                                                          #
# ------------------------------------------------------------------ #
def fijar_semillas(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def buscar_umbral_optimo(y_true, y_prob) -> float:
    """Devuelve el umbral que maximiza F1 sobre la curva PR."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precisions * recalls) / np.maximum(precisions + recalls, 1e-12)
    idx_opt = int(np.argmax(f1[:-1]))
    return float(thresholds[idx_opt])


def split_70_15_15(X: pd.DataFrame, y: pd.Series, seed: int):
    """Split estratificado 70 / 15 / 15."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_SIZE + TEST_SIZE), stratify=y, random_state=seed,
    )
    rel_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------ #
# Busqueda de hiperparametros                                        #
# ------------------------------------------------------------------ #
def hiperparametros_full_search(
    X_train_t, y_train, scale_pos_weight: float, seed: int,
) -> dict:
    """RandomizedSearchCV sobre el espacio del Hito 3."""
    param_dist = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 9),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 11),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 1),
    }
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=SEARCH_N_ITER,
        cv=SEARCH_CV_FOLDS,
        scoring=SEARCH_SCORING,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=2,
        refit=False,
    )
    logger.info("Lanzando RandomizedSearchCV (n_iter=%s, cv=%s).",
                SEARCH_N_ITER, SEARCH_CV_FOLDS)
    search.fit(X_train_t, y_train)
    logger.info("Mejor PR-AUC CV: %.4f", search.best_score_)
    return {"params": search.best_params_, "cv_score": float(search.best_score_)}


# ------------------------------------------------------------------ #
# Pipeline principal                                                  #
# ------------------------------------------------------------------ #
def entrenar(
    dataset_path: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    seed: int,
    search_mode: str,
) -> dict:
    fijar_semillas(seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset(dataset_path)
    X, y = separar_X_y(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(X, y, seed)
    logger.info(
        "Particion: train=%s, val=%s, test=%s (positivos: %d / %d / %d)",
        len(X_train), len(X_val), len(X_test),
        int(y_train.sum()), int(y_val.sum()), int(y_test.sum()),
    )

    preprocesador = construir_preprocesador()
    logger.info("Ajustando preprocesador en TRAIN (sin leakage).")
    X_train_t = preprocesador.fit_transform(X_train)
    X_val_t = preprocesador.transform(X_val)
    X_test_t = preprocesador.transform(X_test)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.2f (calculado en TRAIN).", scale_pos_weight)

    # ----- Busqueda de hiperparametros ----- #
    if search_mode == "full":
        info_busqueda = hiperparametros_full_search(
            X_train_t, y_train, scale_pos_weight, seed,
        )
        mejores_params = info_busqueda["params"]
        cv_score = info_busqueda["cv_score"]
    else:
        logger.info(
            "Modo 'quick': usando hiperparametros validados en el Hito 3 (NB05)."
        )
        mejores_params = dict(BEST_PARAMS_HITO3)
        cv_score = None

    # ----- Refit en train + val ----- #
    logger.info("Refit en train+val con los mejores hiperparametros.")
    X_trainval_t = np.vstack([X_train_t, X_val_t])
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    t0 = time.perf_counter()
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
        **mejores_params,
    )
    modelo.fit(X_trainval_t, y_trainval)
    tiempo_entrenamiento = time.perf_counter() - t0
    logger.info("Modelo entrenado en %.1fs.", tiempo_entrenamiento)

    # ----- Evaluacion sobre test real ----- #
    y_prob = modelo.predict_proba(X_test_t)[:, 1]
    umbral = buscar_umbral_optimo(y_test, y_prob)
    y_pred = (y_prob >= umbral).astype(int)

    metricas = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "umbral_optimo": float(umbral),
    }
    logger.info("Metricas test: %s", json.dumps(metricas, indent=2))

    # ----- Persistencia ----- #
    ruta_modelo = artifacts_dir / "modelo_final.joblib"
    ruta_pre = artifacts_dir / "preprocesador.joblib"
    ruta_meta = artifacts_dir / "metadatos.json"

    joblib.dump(modelo, ruta_modelo)
    joblib.dump(preprocesador, ruta_pre)

    metadatos = {
        "modelo": "XGBoost",
        "notebook_origen": "05_final_model_production.ipynb",
        "contenedor_origen": "guardianai-training",
        "fecha_entrenamiento": datetime.utcnow().isoformat() + "Z",
        "manejo_desbalanceo": "scale_pos_weight",
        "smote": False,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "split": "70/15/15 estratificado",
        "random_state": seed,
        "busqueda": {
            "modo": search_mode,
            "tipo": "RandomizedSearchCV" if search_mode == "full" else "fija (Hito 3)",
            "n_iter": SEARCH_N_ITER if search_mode == "full" else None,
            "cv_folds": SEARCH_CV_FOLDS if search_mode == "full" else None,
            "scoring": SEARCH_SCORING if search_mode == "full" else None,
            "mejor_pr_auc_cv": cv_score,
        },
        "mejores_params": mejores_params,
        "metricas_test": metricas,
        "tiempo_entrenamiento_s": round(tiempo_entrenamiento, 1),
        "columnas_continuas": list(preprocesador.transformers_[0][2]),
        "columnas_categoricas": list(preprocesador.transformers_[1][2]),
    }
    with open(ruta_meta, "w", encoding="utf-8") as fh:
        json.dump(metadatos, fh, indent=2, ensure_ascii=False)

    logger.info("Artefactos persistidos en %s:", artifacts_dir)
    for ruta in (ruta_modelo, ruta_pre, ruta_meta):
        logger.info("  - %s (%.1f KB)", ruta.name, ruta.stat().st_size / 1024)

    # Reporte adicional con el resumen de metricas (legible).
    with open(reports_dir / "metricas_test.json", "w", encoding="utf-8") as fh:
        json.dump(metricas, fh, indent=2, ensure_ascii=False)

    return metadatos


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de GuardianAI (Hito 4).",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        help="Ruta al CSV de entrada (Base.csv).",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=ARTIFACTS_DIR,
        help="Carpeta donde escribir modelo, preprocesador y metadatos.",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help="Carpeta donde escribir reportes de evaluacion.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--search-mode", choices=["quick", "full"], default=SEARCH_MODE,
        help="'quick' usa los hiperparametros del Hito 3; 'full' lanza RandomizedSearchCV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Configuracion: %s", vars(args))
    entrenar(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
        seed=args.seed,
        search_mode=args.search_mode,
    )
    logger.info("Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()
