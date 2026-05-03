"""
schemas.py -- Contratos Pydantic para la API de inferencia.

Define el formato exacto que esperan y devuelven los endpoints. La validacion
en el borde evita que datos malformados lleguen al modelo y traduce errores
de tipo en respuestas HTTP 422 limpias.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ====================================================================== #
# Entrada                                                                 #
# ====================================================================== #
class Transaccion(BaseModel):
    """Una transaccion bancaria del Bank Account Fraud Suite (NeurIPS 2022).

    Coincide 1:1 con las 31 columnas de Base.csv (todas excepto fraud_bool).
    """

    # ---- Continuas ----
    income: float = Field(..., ge=0.0, le=1.0, description="Ingreso normalizado [0, 1].")
    name_email_similarity: float = Field(..., ge=0.0, le=1.0)
    prev_address_months_count: int = Field(..., ge=-1)
    current_address_months_count: int = Field(..., ge=-1)
    customer_age: int = Field(..., ge=0, le=120)
    days_since_request: float = Field(..., ge=0.0)
    intended_balcon_amount: float
    zip_count_4w: int = Field(..., ge=0)
    velocity_6h: float
    velocity_24h: float = Field(..., ge=0.0)
    velocity_4w: float = Field(..., ge=0.0)
    bank_branch_count_8w: int = Field(..., ge=0)
    date_of_birth_distinct_emails_4w: int = Field(..., ge=0)
    bank_months_count: int = Field(..., ge=-1)
    proposed_credit_limit: float = Field(..., ge=0.0)
    session_length_in_minutes: float = Field(..., ge=0.0)
    device_distinct_emails_8w: int = Field(..., ge=-1)
    device_fraud_count: int = Field(..., ge=0)
    month: int = Field(..., ge=0, le=11)

    # ---- Categoricas ----
    payment_type: Literal["AA", "AB", "AC", "AD", "AE"]
    employment_status: Literal["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
    housing_status: Literal["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    source: Literal["INTERNET", "TELEAPP"]
    device_os: Literal["linux", "macintosh", "other", "windows", "x11"]

    # ---- Binarias / passthrough ----
    credit_risk_score: int
    email_is_free: int = Field(..., ge=0, le=1)
    phone_home_valid: int = Field(..., ge=0, le=1)
    phone_mobile_valid: int = Field(..., ge=0, le=1)
    has_other_cards: int = Field(..., ge=0, le=1)
    foreign_request: int = Field(..., ge=0, le=1)
    keep_alive_session: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "income": 0.3,
                "name_email_similarity": 0.45,
                "prev_address_months_count": -1,
                "current_address_months_count": 24,
                "customer_age": 30,
                "days_since_request": 0.012,
                "intended_balcon_amount": -1.0,
                "payment_type": "AB",
                "zip_count_4w": 1500,
                "velocity_6h": 4500.0,
                "velocity_24h": 5200.0,
                "velocity_4w": 4800.0,
                "bank_branch_count_8w": 5,
                "date_of_birth_distinct_emails_4w": 3,
                "employment_status": "CA",
                "credit_risk_score": 150,
                "email_is_free": 1,
                "housing_status": "BC",
                "phone_home_valid": 1,
                "phone_mobile_valid": 1,
                "bank_months_count": 24,
                "has_other_cards": 0,
                "proposed_credit_limit": 1500.0,
                "foreign_request": 0,
                "source": "INTERNET",
                "session_length_in_minutes": 4.7,
                "device_os": "windows",
                "keep_alive_session": 1,
                "device_distinct_emails_8w": 1,
                "device_fraud_count": 0,
                "month": 3,
            }
        }
    }


class LoteTransacciones(BaseModel):
    """Carga util para inferencia en lote (batch)."""

    transacciones: List[Transaccion] = Field(..., min_length=1, max_length=10_000)


# ====================================================================== #
# Salida                                                                  #
# ====================================================================== #
class PrediccionItem(BaseModel):
    probabilidad_fraude: float = Field(..., ge=0.0, le=1.0)
    etiqueta: int = Field(..., ge=0, le=1, description="1 = fraude, 0 = legitima.")
    umbral: float = Field(..., description="Umbral activo aplicado.")
    decision: Literal["PERMITIR", "BLOQUEAR"]
    nivel_riesgo: Literal["BAJO", "MEDIO", "ALTO", "CRITICO"]


class PrediccionResponse(BaseModel):
    status: Literal["success"] = "success"
    model_version: str
    prediccion: PrediccionItem


class PrediccionLoteResponse(BaseModel):
    status: Literal["success"] = "success"
    model_version: str
    n: int
    predicciones: List[PrediccionItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    artifacts_dir: str
    modelo_cargado: bool
    umbral_activo: Optional[float] = None


class MetadataResponse(BaseModel):
    modelo: str
    umbral_activo: float
    fecha_entrenamiento: Optional[str] = None
    metricas_test: dict
    n_columnas_continuas: int
    n_columnas_categoricas: int
