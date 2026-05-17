"""
monitor.py -- Servicio de monitorización GuardianAI (Hito 5).

Responsabilidades:
  1. Metricas operativas: CPU y RAM via psutil. Alerta si superan umbral.
  2. Metricas del modelo: tasa de fraude en predicciones recientes. Alerta si anomala.
  3. Deteccion de drift: compara distribucion de features recientes vs referencia.
     Alerta si alguna feature se desvia mas de DRIFT_Z_THRESHOLD desviaciones tipicas.
  4. Alertas: Telegram.
  5. Feedback loop: si hay drift, dispara reentrenamiento via Docker CLI.

Variables de entorno configurables (ver docker-compose.yml):
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ARTIFACTS_DIR, LOGS_DIR, API_URL,
  CHECK_INTERVAL, CPU_THRESHOLD, RAM_THRESHOLD, DRIFT_Z_THRESHOLD,
  FRAUD_RATE_THRESHOLD, MIN_PREDICTIONS_DRIFT, RETRAIN_ON_DRIFT
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import psutil
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitor")

# ---------------------------------------------------------------------- #
# Configuracion                                                          #
# ---------------------------------------------------------------------- #
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
ARTIFACTS_DIR       = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))
LOGS_DIR            = Path(os.getenv("LOGS_DIR", "/logs"))
API_URL             = os.getenv("API_URL", "http://inference:8000")
CHECK_INTERVAL      = int(os.getenv("CHECK_INTERVAL", "60"))        # segundos
CPU_THRESHOLD       = float(os.getenv("CPU_THRESHOLD", "80.0"))     # %
RAM_THRESHOLD       = float(os.getenv("RAM_THRESHOLD", "80.0"))     # %
DRIFT_Z_THRESHOLD   = float(os.getenv("DRIFT_Z_THRESHOLD", "3.0"))  # desv. tipicas
FRAUD_RATE_THRESHOLD= float(os.getenv("FRAUD_RATE_THRESHOLD", "0.10"))
MIN_PREDICTIONS     = int(os.getenv("MIN_PREDICTIONS_DRIFT", "30"))
RETRAIN_ON_DRIFT    = os.getenv("RETRAIN_ON_DRIFT", "false").lower() == "true"
ALERT_REMINDER_HOURS = float(os.getenv("ALERT_REMINDER_HOURS", "1"))

PREDICTIONS_LOG     = LOGS_DIR / "predictions_log.jsonl"
REFERENCE_STATS     = ARTIFACTS_DIR / "reference_stats.json"
ALERT_STATE_FILE    = LOGS_DIR / "alert_state.json"

# Features numericas a monitorizar para drift
FEATURES_NUMERICAS = [
    "income", "name_email_similarity", "customer_age",
    "days_since_request", "velocity_6h", "velocity_24h",
    "velocity_4w", "zip_count_4w", "proposed_credit_limit",
    "session_length_in_minutes",
]

# Estadisticas de referencia del dataset de entrenamiento (EDA del Hito 2).
# Si el archivo reference_stats.json no existe en /artifacts, se usan estos valores.
REFERENCE_STATS_DEFAULT = {
    "income":                     {"mean": 0.49,   "std": 0.28},
    "name_email_similarity":      {"mean": 0.50,   "std": 0.29},
    "customer_age":               {"mean": 35.0,   "std": 14.0},
    "days_since_request":         {"mean": 3.5,    "std": 8.0},
    "velocity_6h":                {"mean": 4500.0, "std": 2800.0},
    "velocity_24h":               {"mean": 4500.0, "std": 2100.0},
    "velocity_4w":                {"mean": 4500.0, "std": 2000.0},
    "zip_count_4w":               {"mean": 3000.0, "std": 2200.0},
    "proposed_credit_limit":      {"mean": 1000.0, "std": 600.0},
    "session_length_in_minutes":  {"mean": 5.0,    "std": 6.0},
}


# ---------------------------------------------------------------------- #
# Telegram                                                               #
# ---------------------------------------------------------------------- #
def enviar_telegram(mensaje: str) -> None:
    """Envia un mensaje por Telegram. Si no hay token configurado, lo imprime."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("[TELEGRAM-SIM] %s", mensaje)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning("Telegram devolvio error: %s", resp.text)
    except Exception as exc:
        logger.warning("Error enviando Telegram: %s", exc)


# ---------------------------------------------------------------------- #
# 1. Metricas operativas                                                 #
# ---------------------------------------------------------------------- #
def comprobar_metricas_operativas() -> Tuple[float, float, List[str]]:
    """Devuelve (cpu%, ram%, lista_alertas)."""
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    alertas = []
    if cpu > CPU_THRESHOLD:
        alertas.append(f"CPU alta: <b>{cpu:.1f}%</b> (umbral: {CPU_THRESHOLD}%)")
    if ram > RAM_THRESHOLD:
        alertas.append(f"RAM alta: <b>{ram:.1f}%</b> (umbral: {RAM_THRESHOLD}%)")
    return cpu, ram, alertas


# ---------------------------------------------------------------------- #
# 2. Carga del log de predicciones                                       #
# ---------------------------------------------------------------------- #
def cargar_predicciones(ultimas_n: int = 500) -> list:
    """Devuelve las ultimas N predicciones del log JSONL."""
    if not PREDICTIONS_LOG.exists():
        return []
    registros = []
    try:
        with open(PREDICTIONS_LOG, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        registros.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.warning("Error leyendo log de predicciones: %s", exc)
    return registros[-ultimas_n:]


# ---------------------------------------------------------------------- #
# 3. Tasa de fraude                                                      #
# ---------------------------------------------------------------------- #
def comprobar_tasa_fraude(registros: list) -> Tuple[float | None, List[str]]:
    """Alerta si la tasa de fraude reciente supera FRAUD_RATE_THRESHOLD."""
    if len(registros) < MIN_PREDICTIONS:
        return None, []
    tasa = sum(r.get("etiqueta", 0) for r in registros) / len(registros)
    alertas = []
    if tasa > FRAUD_RATE_THRESHOLD:
        alertas.append(
            f"Tasa de fraude anomala: <b>{tasa*100:.1f}%</b> "
            f"(umbral: {FRAUD_RATE_THRESHOLD*100:.0f}%)\n"
            f"Sobre {len(registros)} predicciones recientes."
        )
    return tasa, alertas


# ---------------------------------------------------------------------- #
# 4. Deteccion de drift                                                  #
# ---------------------------------------------------------------------- #
def _cargar_referencia() -> dict:
    """Carga estadisticas de referencia desde archivo o usa las por defecto."""
    if REFERENCE_STATS.exists():
        try:
            with open(REFERENCE_STATS, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return REFERENCE_STATS_DEFAULT


def comprobar_drift(registros: list) -> Tuple[bool, List[str]]:
    """
    Compara la media de cada feature numerica en las predicciones recientes
    contra las estadisticas de referencia del entrenamiento.
    Drift = desviacion de mas de DRIFT_Z_THRESHOLD desviaciones tipicas.
    """
    if len(registros) < MIN_PREDICTIONS:
        return False, []

    referencia = _cargar_referencia()
    features_con_drift = []

    for feature in FEATURES_NUMERICAS:
        if feature not in referencia:
            continue
        ref_mean = referencia[feature]["mean"]
        ref_std  = referencia[feature]["std"]
        if ref_std == 0:
            continue

        valores = [r[feature] for r in registros if feature in r]
        if not valores:
            continue

        media_actual = sum(valores) / len(valores)
        z_score = abs(media_actual - ref_mean) / ref_std

        if z_score > DRIFT_Z_THRESHOLD:
            features_con_drift.append(
                f"  • {feature}: media={media_actual:.2f} "
                f"(ref={ref_mean:.2f}, z={z_score:.1f})"
            )

    if features_con_drift:
        detalle = "\n".join(features_con_drift)
        alertas = [
            f"<b>Data drift detectado</b> en {len(features_con_drift)} feature(s):\n"
            f"{detalle}"
        ]
        return True, alertas

    return False, []


# ---------------------------------------------------------------------- #
# 5. Feedback loop: reentrenamiento                                      #
# ---------------------------------------------------------------------- #
def disparar_reentrenamiento() -> None:
    """Lanza el contenedor de entrenamiento via Docker CLI."""
    enviar_telegram("🔄 <b>Iniciando reentrenamiento automatico...</b>")
    logger.info("Disparando reentrenamiento via docker compose.")
    try:
        result = subprocess.run(
            [
                "docker", "compose",
                "--profile", "train",
                "run", "--rm", "training",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            enviar_telegram("<b>Reentrenamiento completado</b> correctamente.")
            logger.info("Reentrenamiento completado.")
        else:
            msg = result.stderr[:400] if result.stderr else "sin detalles"
            enviar_telegram(f"<b>Error en reentrenamiento:</b>\n<code>{msg}</code>")
            logger.error("Error en reentrenamiento: %s", result.stderr)
    except subprocess.TimeoutExpired:
        enviar_telegram("Reentrenamiento agotó el tiempo límite (1h).")
    except Exception as exc:
        enviar_telegram(f"Error al lanzar reentrenamiento: {exc}")
        logger.error("Error al disparar reentrenamiento: %s", exc)

# ---------------------------------------------------------------------- #
# Estado de alertas (deduplicacion)                                      #
# ---------------------------------------------------------------------- #
def leer_estado_alerta() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            with open(ALERT_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"drift": False, "fraud_rate": False, "cpu": False, "ram": False, "ultima_alerta": None}


def guardar_estado_alerta(estado: dict) -> None:
    try:
        with open(ALERT_STATE_FILE, "w") as f:
            json.dump(estado, f)
    except Exception as exc:
        logger.warning("No se pudo guardar estado de alerta: %s", exc)

# ---------------------------------------------------------------------- #
# Bucle principal                                                        #
# ---------------------------------------------------------------------- #
def ciclo_monitorizacion() -> None:
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Metricas operativas
    cpu, ram, alertas_op = comprobar_metricas_operativas()

    # 2. Cargar predicciones recientes
    registros = cargar_predicciones(ultimas_n=500)

    # 3. Tasa de fraude
    tasa_fraude, alertas_fraude = comprobar_tasa_fraude(registros)

    # 4. Drift
    drift_detectado, alertas_drift = comprobar_drift(registros)

    # 5. Log de estado
    tasa_str = f"{tasa_fraude*100:.1f}%" if tasa_fraude is not None else "N/A"
    logger.info(
        "[%s] CPU=%.1f%% RAM=%.1f%% Predicciones=%d TasaFraude=%s Drift=%s",
        ahora, cpu, ram, len(registros), tasa_str, drift_detectado,
    )

     # 6. Enviar alertas consolidadas solo si el estado cambia
    todas_alertas = alertas_op + alertas_fraude + alertas_drift

    estado_anterior = leer_estado_alerta()
    estado_actual = {
        "drift": drift_detectado,
        "fraud_rate": len(alertas_fraude) > 0,
        "cpu": len([a for a in alertas_op if "CPU" in a]) > 0,
        "ram": len([a for a in alertas_op if "RAM" in a]) > 0,
    }

    estado_limpio = {"drift": False, "fraud_rate": False, "cpu": False, "ram": False}

    ahora_dt = datetime.utcnow()
    ultima_alerta_str = estado_anterior.get("ultima_alerta")
    horas_desde_ultima = float("inf")
    if ultima_alerta_str:
        try:
            ultima_alerta_dt = datetime.fromisoformat(ultima_alerta_str)
            horas_desde_ultima = (ahora_dt - ultima_alerta_dt).total_seconds() / 3600
        except Exception:
            pass

    campos = ["drift", "fraud_rate", "cpu", "ram"]
    hay_cambio = any(estado_actual.get(c) != estado_anterior.get(c) for c in campos)
    hay_recordatorio = todas_alertas and horas_desde_ultima >= ALERT_REMINDER_HOURS

    if hay_cambio or hay_recordatorio:
        if todas_alertas:
            prefijo = "<b>Alerta |</b>" if hay_cambio else "<b>Recordatorio |</b> <i>Recordatorio</i>"
            cuerpo = "\n\n".join(todas_alertas)
            mensaje = (
                f"{prefijo} <b>GuardianAI Monitor</b> [{ahora}]\n\n"
                f"{cuerpo}"
            )
            enviar_telegram(mensaje)
            estado_actual["ultima_alerta"] = ahora_dt.isoformat()
        elif hay_cambio and estado_anterior != estado_limpio:
            enviar_telegram(
                f"<b>GuardianAI Monitor</b> [{ahora}]\n"
                f"El sistema ha vuelto a estado normal."
            )
            estado_actual["ultima_alerta"] = None
        guardar_estado_alerta(estado_actual)

    # 7. Feedback loop
    if drift_detectado and RETRAIN_ON_DRIFT:
        disparar_reentrenamiento()


def main() -> None:
    logger.info("Monitor GuardianAI iniciado.")
    logger.info(
        "Config: CHECK_INTERVAL=%ds CPU_TH=%.0f%% RAM_TH=%.0f%% "
        "DRIFT_Z=%.1f FRAUD_TH=%.0f%% RETRAIN=%s",
        CHECK_INTERVAL, CPU_THRESHOLD, RAM_THRESHOLD,
        DRIFT_Z_THRESHOLD, FRAUD_RATE_THRESHOLD * 100, RETRAIN_ON_DRIFT,
    )

    enviar_telegram(
        "<b>GuardianAI Monitor iniciado</b>\n"
        f"Intervalo de comprobacion: {CHECK_INTERVAL}s\n"
        f"Drift z-threshold: {DRIFT_Z_THRESHOLD}\n"
        f"Reentrenamiento automatico: {'SI' if RETRAIN_ON_DRIFT else 'NO'}"
    )

    retrain_en_curso = False

    while True:
        try:
            ciclo_monitorizacion()
        except Exception as exc:
            logger.error("Error inesperado en ciclo de monitorizacion: %s", exc)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
