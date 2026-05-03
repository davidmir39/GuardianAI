"""
config.py -- Configuracion del contenedor de entrenamiento (Hito 4).

Reproduce las constantes relevantes del notebook 05_final_model_production
para que el contenedor sea autocontenido (no depende de src/ del repo).
Las rutas son siempre configurables por variable de entorno, siguiendo la
regla del Hito 4: "Las rutas siempre seran configurables por comandos o
variables de entorno".
"""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------ #
# Rutas (configurables por entorno)                                  #
# ------------------------------------------------------------------ #
# Dataset crudo (Base.csv del Bank Account Fraud Suite, NeurIPS 2022).
DATASET_PATH = Path(os.getenv("DATASET_PATH", "/data/raw/Base.csv"))

# Carpeta donde se persisten modelo, preprocesador y metadatos.
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))

# Carpeta donde se persisten reportes de evaluacion (metricas, curvas).
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/artifacts/reports"))

# ------------------------------------------------------------------ #
# Reproducibilidad y particion                                       #
# ------------------------------------------------------------------ #
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
TARGET_COL = "fraud_bool"

# ------------------------------------------------------------------ #
# Busqueda de hiperparametros                                        #
# ------------------------------------------------------------------ #
# Modo rapido (por defecto): usa los mejores hiperparametros encontrados
# en el Hito 3 (metadatos.json) y solo reentrena. Util para validar la
# tuberia de despliegue end-to-end en minutos.
# Modo completo: ejecuta RandomizedSearchCV (n_iter, cv).
SEARCH_MODE = os.getenv("SEARCH_MODE", "quick").lower()  # 'quick' | 'full'
SEARCH_N_ITER = int(os.getenv("SEARCH_N_ITER", "30"))
SEARCH_CV_FOLDS = int(os.getenv("SEARCH_CV_FOLDS", "3"))
SEARCH_SCORING = os.getenv("SEARCH_SCORING", "average_precision")
N_JOBS = int(os.getenv("N_JOBS", "-1"))

# Hiperparametros validados en el Hito 3 (NB05). Se usan en modo 'quick'.
BEST_PARAMS_HITO3 = {
    "colsample_bytree": 0.9933692563579372,
    "learning_rate": 0.08577664406446507,
    "max_depth": 4,
    "min_child_weight": 9,
    "n_estimators": 223,
    "reg_alpha": 0.5081987767407187,
    "reg_lambda": 0.6958128067908819,
    "subsample": 0.943343521925488,
}

# ------------------------------------------------------------------ #
# Esquema de columnas (debe coincidir con el preprocesador)          #
# ------------------------------------------------------------------ #
COLUMNAS_CONTINUAS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "bank_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
]

COLUMNAS_CATEGORICAS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Resto de columnas (passthrough en el ColumnTransformer): binarias/enteras.
COLUMNAS_PASSTHROUGH = [
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
]

# Mapa de tipos para una carga eficiente (~50% memoria frente a default).
DTYPE_BASE_CSV = {
    "fraud_bool": "int8",
    "income": "float32",
    "name_email_similarity": "float32",
    "prev_address_months_count": "int16",
    "current_address_months_count": "int16",
    "customer_age": "int8",
    "days_since_request": "float32",
    "intended_balcon_amount": "float32",
    "payment_type": "category",
    "zip_count_4w": "int32",
    "velocity_6h": "float32",
    "velocity_24h": "float32",
    "velocity_4w": "float32",
    "bank_branch_count_8w": "int32",
    "date_of_birth_distinct_emails_4w": "int16",
    "employment_status": "category",
    "credit_risk_score": "int16",
    "email_is_free": "int8",
    "housing_status": "category",
    "phone_home_valid": "int8",
    "phone_mobile_valid": "int8",
    "bank_months_count": "int16",
    "has_other_cards": "int8",
    "proposed_credit_limit": "float32",
    "foreign_request": "int8",
    "source": "category",
    "session_length_in_minutes": "float32",
    "device_os": "category",
    "keep_alive_session": "int8",
    "device_distinct_emails_8w": "int16",
    "device_fraud_count": "int8",
    "month": "int8",
}
