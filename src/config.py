"""
config.py — Configuracion central del proyecto GuardianAI (Hito 3).

Este modulo centraliza constantes que se usan en varios notebooks para
garantizar reproducibilidad y coherencia entre ellos.
"""

import os
import random
import numpy as np

# ------------------------------------------------------------------ #
# Rutas                                                               #
# ------------------------------------------------------------------ #
# Ruta relativa desde `src/` hacia la raiz del repositorio.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "Base.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # excluido en .gitignore

# ------------------------------------------------------------------ #
# Reproducibilidad                                                    #
# ------------------------------------------------------------------ #
RANDOM_STATE = 42

# ------------------------------------------------------------------ #
# Particion de datos                                                  #
# ------------------------------------------------------------------ #
# Split 70 / 15 / 15 (train / val / test) con estratificacion sobre `fraud_bool`.
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ------------------------------------------------------------------ #
# Manejo del desbalanceo                                              #
# ------------------------------------------------------------------ #
# Prevalencia del fraude en Bank Account Fraud Suite (NeurIPS 2022): ~1.1 %.
# `scale_pos_weight` se calcula dinamicamente en cada notebook como
# `(N_negativos / N_positivos)` en el TRAIN, evitando leakage.

# ------------------------------------------------------------------ #
# Recursos de computo                                                 #
# ------------------------------------------------------------------ #
N_JOBS = -1  # usa todos los nucleos disponibles

# ------------------------------------------------------------------ #
# Variables continuas del dataset (para StandardScaler selectivo)    #
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

# ------------------------------------------------------------------ #
# dtype map para carga eficiente de Base.csv                          #
# ------------------------------------------------------------------ #
# Reduce el consumo de memoria ~50 % frente a la carga por defecto.
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


# ------------------------------------------------------------------ #
# Utilidad: semillas globales                                         #
# ------------------------------------------------------------------ #
def set_global_seeds(seed: int = RANDOM_STATE) -> None:
    """Fija semillas en numpy, random y (si estan disponibles) TensorFlow y PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
