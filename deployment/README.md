# GuardianAI - Despliegue (Hito 4 + Hito 5)

API REST de deteccion de fraude bancario en tiempo real, empaquetada en
contenedores Docker y orquestada con Docker Compose. Incluye sistema de
monitorizacion, deteccion de drift y alertas por Telegram (Hito 5).

> **TL;DR (3 comandos)**
> ```bash
> cd deployment/
> docker compose up
> curl -X POST http://localhost:8000/predict \
>      -H "Content-Type: application/json" \
>      -d @data/samples/transaccion_legitima.json
> ```

---

## Que hay aqui

```
deployment/
├── docker-compose.yml         <- orquesta los 5 servicios (Hito 4 + 5)
├── .env.example               <- variables opcionales (copialo a .env)
├── README.md                  <- esta guia (plug-and-play)
├── DOCUMENTACION.md           <- explicacion detallada de decisiones
├── simulate_anomaly.py        <- script de simulacion de entorno anomalo (Hito 5)
├── training/                  <- contenedor de entrenamiento
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pipeline/
│       ├── config.py          <- rutas, columnas, hiperparametros
│       ├── data_ingestion.py  <- carga de Base.csv
│       ├── features.py        <- ColumnTransformer (scaler + one-hot)
│       └── train.py           <- orquestador end-to-end
├── inference/                 <- contenedor de inferencia (API)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── api/
│       ├── settings.py        <- variables de entorno
│       ├── predictor.py       <- wrapper de carga + predict
│       ├── schemas.py         <- contratos Pydantic
│       └── main.py            <- aplicacion FastAPI + instrumentacion Prometheus
├── monitoring/                <- contenedor de monitorizacion (Hito 5)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── monitor.py             <- deteccion de drift + alertas + feedback loop
│   └── prometheus.yml         <- configuracion de Prometheus
├── shared/
│   ├── artifacts/             <- volumen compartido (modelo + preprocesador)
│   └── logs/                  <- logs de predicciones + estado de alertas (Hito 5)
└── data/
    ├── raw/                   <- aqui va Base.csv si quiere reentrenar
    └── samples/               <- JSONs de ejemplo para probar la API
```

---

## Requisitos previos

- Docker Desktop >= 4.30 (o Docker Engine + Docker Compose v2).
- (Opcional) `curl` o Postman para probar la API.
- (Opcional) Python con `requests` instalado para ejecutar `simulate_anomaly.py`.
- (Opcional, solo si reentrena) [Base.csv del Bank Account Fraud Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).
- (Hito 5) Bot de Telegram — ver seccion de configuracion mas abajo.

No se necesita Python ni instalar dependencias en el host: todo corre en Docker.

---

## Arranque plug-and-play (no requiere Base.csv)

Levanta la API usando el modelo ya entrenado en el Hito 3:

```bash
cd deployment/
docker compose up
```

Que pasa por debajo:

1. Se construye la imagen `guardianai-inference` y `guardianai-monitor`.
2. El servicio `bootstrap` (one-shot) copia los artefactos de `../models/`
   (`modelo_final.joblib`, `preprocesador.joblib`, `metadatos.json`) al
   volumen compartido `./shared/artifacts/`.
3. El servicio `inference` arranca FastAPI + uvicorn en el puerto 8000.
4. El servicio `prometheus` empieza a raspar metricas de la API cada 15s.
5. El servicio `monitor` arranca y comienza a comprobar metricas cada 60s.

Cuando vea el log `Application startup complete.`, abra:

- Documentacion interactiva: <http://localhost:8000/docs>
- Comprobacion de salud: <http://localhost:8000/health>
- Metadatos del modelo: <http://localhost:8000/metadata>
- Metricas Prometheus: <http://localhost:8000/metrics>
- Panel Prometheus: <http://localhost:9090>

Para detenerlo: `Ctrl + C` y luego `docker compose down`.

---

## Probar la API

### Una transaccion

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @data/samples/transaccion_legitima.json
```

Respuesta esperada (ejemplo):

```json
{
  "status": "success",
  "model_version": "XGBoost",
  "prediccion": {
    "probabilidad_fraude": 0.0123,
    "etiqueta": 0,
    "umbral": 0.9177,
    "decision": "PERMITIR",
    "nivel_riesgo": "BAJO"
  }
}
```

### Lote de transacciones

```bash
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d @data/samples/lote_ejemplo.json
```

### Probar una transaccion sospechosa

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @data/samples/transaccion_sospechosa.json
```

---

## Hito 5: Monitorizacion y alertas

### Configuracion del bot de Telegram

Antes de arrancar, es necesario configurar un bot de Telegram para recibir alertas:

1. Abre Telegram y busca **@BotFather**
2. Escribe `/newbot` y sigue las instrucciones — te dara un **token**
3. Busca **@userinfobot** y escribele cualquier cosa — te dara tu **chat_id**
4. Escribe al bot al menos una vez (cualquier mensaje) para desbloquear el canal
5. Copia `.env.example` a `.env` y rellena:

```
TELEGRAM_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui
```

### Que monitoriza el sistema

**Metricas operativas** (Prometheus + psutil):
- CPU y RAM del sistema
- Peticiones por segundo y latencia (via `/metrics`)

**Metricas del modelo**:
- Tasa de fraude en las predicciones recientes
- Alerta si supera `FRAUD_RATE_THRESHOLD` (por defecto 10%)

**Deteccion de drift**:
- Compara la distribucion de features de las predicciones recientes
  contra las estadisticas de referencia del entrenamiento
- Alerta si alguna feature se desvia mas de `DRIFT_Z_THRESHOLD` desviaciones tipicas (por defecto 3.0)

### Sistema de alertas

El monitor implementa deduplicacion inteligente:

- **Primera deteccion** → alerta inmediata 
- **Problema persiste** → recordatorio cada `ALERT_REMINDER_HOURS` horas 
- **Problema resuelto** → notificacion de recuperacion 
- **No spamea** — no repite la misma alerta en cada ciclo

### Simular un entorno anomalo

Para demostrar el sistema de alertas, instala `requests` y ejecuta:

```bash
pip install requests
python simulate_anomaly.py --n 100 --modo fraude
```

Modos disponibles:
- `fraude` — solicitudes con caracteristicas tipicas de fraude masivo
- `normal` — solicitudes tipicas de cliente legitimo
- `mixto` — 80% fraude + 20% normal (por defecto)

En menos de 60 segundos deberia llegar una alerta por Telegram con el drift detectado.

### Limpiar el estado entre pruebas

Si quiere resetear el sistema para una nueva demostracion:

**En Windows (PowerShell):**
```powershell
[System.IO.File]::WriteAllText("shared\logs\predictions_log.jsonl", "")
[System.IO.File]::WriteAllText("shared\logs\alert_state.json", "")
```

**En Linux/Mac:**
```bash
> shared/logs/predictions_log.jsonl
> shared/logs/alert_state.json
```

### Feedback loop: reentrenamiento automatico

Para activar el reentrenamiento automatico cuando se detecte drift,
cambie en `.env`:

```
RETRAIN_ON_DRIFT=true
```

Y asegurese de tener `Base.csv` en `deployment/data/raw/`. El contenedor
de monitor lanzara el reentrenamiento automaticamente y le notificara
el resultado por Telegram.

---

## Re-entrenar el modelo (opcional)

Si quiere generar un modelo nuevo en lugar de usar el preentrenado:

1. Descargue `Base.csv` y coloquelo en `deployment/data/raw/Base.csv`.
2. Ejecute:

   ```bash
   docker compose --profile train run --rm training
   ```

3. Para hacer una busqueda completa de hiperparametros (~30-60 min):

   ```bash
   SEARCH_MODE=full docker compose --profile train run --rm training
   ```

4. Recargue la API para que tome el nuevo modelo:

   ```bash
   docker compose restart inference
   ```

---

## Endpoints de la API

| Metodo | Ruta              | Descripcion                                       |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Redirige a `/docs` (Swagger UI).                  |
| GET    | `/docs`           | Documentacion interactiva (auto-generada).        |
| GET    | `/health`         | Salud del servicio (modelo cargado, umbral...).   |
| GET    | `/metadata`       | Metadatos del modelo (metricas test, fecha, ...). |
| GET    | `/metrics`        | Metricas Prometheus (HTTP, CPU, RAM, fraude).     |
| POST   | `/predict`        | Prediccion para 1 transaccion.                    |
| POST   | `/predict/batch`  | Prediccion para hasta 10.000 transacciones.       |

Codigos de error tipicos:

- `422 Unprocessable Entity` -- el JSON no respeta el contrato (`schemas.py`).
- `503 Service Unavailable` -- los artefactos no estan cargados.
- `400 Bad Request` -- error en el `predict` (preprocesador, modelo).

---

## Variables de entorno

Copie `.env.example` a `.env` y modifique lo que necesite.

### Hito 4 — API e inferencia

| Variable             | Default | Descripcion                                  |
|----------------------|---------|----------------------------------------------|
| `API_PORT`           | `8000`  | Puerto host para la API.                     |
| `UVICORN_WORKERS`    | `1`     | Procesos uvicorn.                            |
| `THRESHOLD_OVERRIDE` | (vacio) | Forzar un umbral fijo ignorando metadatos.   |
| `DEFAULT_THRESHOLD`  | `0.5`   | Fallback si los metadatos no traen umbral.   |
| `SEARCH_MODE`        | `quick` | `quick` o `full` (RandomizedSearchCV).       |
| `SEARCH_N_ITER`      | `30`    | Iteraciones de la busqueda en modo `full`.   |
| `RANDOM_STATE`       | `42`    | Semilla para reproducibilidad.               |

### Hito 5 — Monitorizacion

| Variable                 | Default | Descripcion                                       |
|--------------------------|---------|---------------------------------------------------|
| `TELEGRAM_TOKEN`         | —       | Token del bot de Telegram.                        |
| `TELEGRAM_CHAT_ID`       | —       | ID del chat donde llegan las alertas.             |
| `CHECK_INTERVAL`         | `60`    | Segundos entre comprobaciones del monitor.        |
| `CPU_THRESHOLD`          | `80.0`  | % CPU para alerta operativa.                      |
| `RAM_THRESHOLD`          | `80.0`  | % RAM para alerta operativa.                      |
| `DRIFT_Z_THRESHOLD`      | `3.0`   | Desviaciones tipicas para detectar drift.         |
| `FRAUD_RATE_THRESHOLD`   | `0.10`  | Tasa de fraude anomala (>10% dispara alerta).     |
| `MIN_PREDICTIONS_DRIFT`  | `30`    | Minimo de predicciones para analizar drift.       |
| `ALERT_REMINDER_HOURS`   | `1`     | Horas entre recordatorios si el problema persiste.|
| `RETRAIN_ON_DRIFT`       | `false` | Reentrenamiento automatico al detectar drift.     |

---

## Solucion de problemas

**`Cannot connect to the Docker daemon`**: arranque Docker Desktop antes
de ejecutar los comandos.

**`bind source path does not exist: ../models`**: ejecute `docker compose`
desde la carpeta `deployment/`, no desde la raiz del repo.

**`503 Service Unavailable` al llamar a `/predict`**: el contenedor
`bootstrap` no termino bien o `../models/` esta vacio. Compruebe:
`docker compose logs bootstrap`.

**`docker: Error response from daemon: ports are not available`**: cambie
`API_PORT` en `.env` por un puerto libre, p. ej. `API_PORT=8080`.

**Las alertas de Telegram no llegan**: asegurese de haber escrito al bot
al menos una vez desde Telegram antes de arrancar el sistema.

**El monitor muestra `Predicciones=0`**: compruebe que la carpeta
`shared/logs/` existe y que el contenedor `inference` tiene el volumen
montado correctamente.

**Quiere parar y limpiar todo**:
```bash
docker compose down --volumes --remove-orphans
docker image rm guardianai-inference guardianai-training guardianai-monitor 2>/dev/null
```

---

## Para mas detalle

Vea [DOCUMENTACION.md](DOCUMENTACION.md) -- explicacion detallada del por
que de cada decision: division en modulos, eleccion de FastAPI, patron
del volumen compartido, modos de entrenamiento, etc.
