"""
simulate_anomaly.py -- Simulacion de entorno anomalo (Hito 5).

Envia rafagas de solicitudes con caracteristicas tipicas de fraude masivo
para disparar las alertas de drift y tasa de fraude del monitor.

Uso:
    python simulate_anomaly.py [--url URL] [--n N] [--modo MODO]

Modos:
    fraude    -- solicitudes con alta probabilidad de fraude (velocidades altas,
                 foreign_request=1, keep_alive_session=1, etc.)
    normal    -- solicitudes tipicas de un cliente legitimo
    mixto     -- mezcla 80% fraude + 20% normal (por defecto)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import requests

# ---------------------------------------------------------------------- #
# Plantillas de solicitudes                                              #
# ---------------------------------------------------------------------- #
SOLICITUD_NORMAL = {
    "income": 0.5,
    "name_email_similarity": 0.6,
    "prev_address_months_count": 24,
    "current_address_months_count": 36,
    "customer_age": 35,
    "days_since_request": 1.0,
    "intended_balcon_amount": -1.0,
    "payment_type": "AB",
    "zip_count_4w": 1500,
    "velocity_6h": 3000.0,
    "velocity_24h": 3500.0,
    "velocity_4w": 4000.0,
    "bank_branch_count_8w": 3,
    "date_of_birth_distinct_emails_4w": 1,
    "employment_status": "CA",
    "credit_risk_score": 100,
    "email_is_free": 0,
    "housing_status": "BC",
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "bank_months_count": 24,
    "has_other_cards": 1,
    "proposed_credit_limit": 800.0,
    "foreign_request": 0,
    "source": "INTERNET",
    "session_length_in_minutes": 5.0,
    "device_os": "windows",
    "keep_alive_session": 0,
    "device_distinct_emails_8w": 1,
    "device_fraud_count": 0,
    "month": 3,
}

SOLICITUD_FRAUDE = {
    "income": 0.9,                         # ingresos muy altos declarados
    "name_email_similarity": 0.05,         # email no relacionado con nombre
    "prev_address_months_count": -1,       # sin historial de direccion anterior
    "current_address_months_count": 1,     # recien llegado
    "customer_age": 22,
    "days_since_request": 0.001,           # solicitud casi instantanea
    "intended_balcon_amount": 5000.0,      # importe muy alto
    "payment_type": "AD",
    "zip_count_4w": 14000,                 # muchisimas solicitudes del mismo CP
    "velocity_6h": 15000.0,               # velocidad extremadamente alta
    "velocity_24h": 14000.0,
    "velocity_4w": 13000.0,
    "bank_branch_count_8w": 1,
    "date_of_birth_distinct_emails_4w": 8, # muchos emails distintos
    "employment_status": "CB",
    "credit_risk_score": 200,
    "email_is_free": 1,                    # email gratuito
    "housing_status": "BD",
    "phone_home_valid": 0,                 # telefono fijo invalido
    "phone_mobile_valid": 1,
    "bank_months_count": -1,               # sin historial bancario
    "has_other_cards": 0,
    "proposed_credit_limit": 4500.0,       # limite muy alto solicitado
    "foreign_request": 1,                  # solicitud desde el extranjero
    "source": "INTERNET",
    "session_length_in_minutes": 0.1,      # sesion casi instantanea (bot)
    "device_os": "linux",
    "keep_alive_session": 1,               # sesion activa persistente
    "device_distinct_emails_8w": 5,
    "device_fraud_count": 2,
    "month": 3,
}


def _variacion(base: dict, ruido: float = 0.1) -> dict:
    """Anade pequeña variacion aleatoria a los campos numericos."""
    out = dict(base)
    campos_float = [
        "income", "name_email_similarity", "days_since_request",
        "intended_balcon_amount", "velocity_6h", "velocity_24h",
        "velocity_4w", "proposed_credit_limit", "session_length_in_minutes",
    ]
    for campo in campos_float:
        if campo in out and isinstance(out[campo], float):
            out[campo] = max(0.0, out[campo] * (1 + random.uniform(-ruido, ruido)))
    return out


def enviar_solicitud(url: str, datos: dict) -> dict | None:
    try:
        resp = requests.post(f"{url}/predict", json=datos, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"  [ERROR] {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulacion de entorno anomalo GuardianAI")
    parser.add_argument("--url",  default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--n",    type=int, default=100, help="Numero de solicitudes a enviar")
    parser.add_argument("--modo", choices=["fraude", "normal", "mixto"], default="mixto")
    parser.add_argument("--delay", type=float, default=0.1, help="Segundos entre solicitudes")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  GuardianAI -- Simulacion de entorno anomalo (Hito 5)")
    print(f"{'='*60}")
    print(f"  URL:   {args.url}")
    print(f"  N:     {args.n} solicitudes")
    print(f"  Modo:  {args.modo}")
    print(f"  Delay: {args.delay}s entre solicitudes")
    print(f"{'='*60}\n")

    # Verificar que la API esta disponible
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        health = resp.json()
        if not health.get("modelo_cargado"):
            print("ERROR: La API no tiene el modelo cargado. Levanta el servicio primero.")
            sys.exit(1)
        print(f"API OK -- umbral activo: {health.get('umbral_activo')}\n")
    except Exception as exc:
        print(f"ERROR: No se puede conectar a la API ({exc}). ¿Esta levantada?")
        sys.exit(1)

    contadores = {"BLOQUEAR": 0, "PERMITIR": 0, "error": 0}

    for i in range(1, args.n + 1):
        # Seleccionar plantilla segun modo
        if args.modo == "fraude":
            datos = _variacion(SOLICITUD_FRAUDE)
        elif args.modo == "normal":
            datos = _variacion(SOLICITUD_NORMAL)
        else:  # mixto: 80% fraude, 20% normal
            datos = _variacion(SOLICITUD_FRAUDE if random.random() < 0.8 else SOLICITUD_NORMAL)

        resultado = enviar_solicitud(args.url, datos)

        if resultado is None:
            contadores["error"] += 1
        else:
            pred = resultado.get("prediccion", {})
            decision = pred.get("decision", "?")
            prob = pred.get("probabilidad_fraude", 0)
            riesgo = pred.get("nivel_riesgo", "?")
            contadores[decision] = contadores.get(decision, 0) + 1
            print(
                f"  [{i:04d}] prob={prob:.4f}  decision={decision}  "
                f"riesgo={riesgo}"
            )

        time.sleep(args.delay)

    # Resumen
    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    total = args.n
    bloqueadas = contadores.get("BLOQUEAR", 0)
    permitidas  = contadores.get("PERMITIR", 0)
    errores     = contadores.get("error", 0)
    print(f"  Total enviadas:  {total}")
    print(f"  BLOQUEAR:        {bloqueadas} ({bloqueadas/total*100:.1f}%)")
    print(f"  PERMITIR:        {permitidas}  ({permitidas/total*100:.1f}%)")
    print(f"  Errores:         {errores}")
    print(f"\n  El monitor deberia detectar anomalia en el proximo ciclo.")
    print(f"  Revisa Telegram y los logs del contenedor 'monitor'.\n")


if __name__ == "__main__":
    main()
