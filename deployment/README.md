# GuardianAI - Despliegue (Hito 4)

API REST de deteccion de fraude bancario en tiempo real, empaquetada en
contenedores Docker y orquestada con Docker Compose.

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
├── docker-compose.yml         <- orquesta los 3 servicios
├── .env.example               <- variables opcionales (copialo a .env)
├── README.md                  <- esta guia (plug-and-play)
├── DOCUMENTACION.md           <- explicacion detallada de decisiones
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
│       └── main.py            <- aplicacion FastAPI
├── shared/artifacts/          <- volumen compartido (modelo + preprocesador)
└── data/
    ├── raw/                   <- aqui va Base.csv si quiere reentrenar
    └── samples/               <- JSONs de ejemplo para probar la API
```

---

## Requisitos previos

- Docker Desktop >= 4.30 (o Docker Engine + Docker Compose v2).
- (Opcional) `curl` o Postman para probar la API.
- (Opcional, solo si reentrena) [Base.csv del Bank Account Fraud Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

No se necesita Python ni instalar dependencias en el host: todo corre en Docker.

---

## Arranque plug-and-play (no requiere Base.csv)

Levanta la API usando el modelo ya entrenado en el Hito 3:

```bash
cd deployment/
docker compose up
```

Que pasa por debajo:

1. Se construye la imagen `guardianai-inference`.
2. El servicio `bootstrap` (one-shot) copia los artefactos de `../models/`
   (`modelo_final.joblib`, `preprocesador.joblib`, `metadatos.json`) al
   volumen compartido `./shared/artifacts/`.
3. El servicio `inference` arranca FastAPI + uvicorn en el puerto 8000.

Cuando vea el log `Application startup complete.`, abra:

- Documentacion interactiva: <http://localhost:8000/docs>
- Comprobacion de salud: <http://localhost:8000/health>
- Metadatos del modelo: <http://localhost:8000/metadata>

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

## Re-entrenar el modelo (opcional)

Si quiere generar un modelo nuevo en lugar de usar el preentrenado:

1. Descargue `Base.csv` y coloquelo en `deployment/data/raw/Base.csv`.
2. Ejecute:

   ```bash
   docker compose --profile train run --rm training
   ```

   Esto lanza el pipeline completo: ingesta -> split 70/15/15 -> preprocesador
   -> XGBoost -> umbral optimo -> persistencia. Por defecto usa los
   hiperparametros validados en el Hito 3 (modo `quick`, ~1 min).

3. Para hacer una busqueda completa de hiperparametros (lenta, ~30-60 min):

   ```bash
   SEARCH_MODE=full docker compose --profile train run --rm training
   ```

4. Recargue la API para que tome el nuevo modelo:

   ```bash
   docker compose restart inference
   ```

Los nuevos artefactos sobreescriben los anteriores en `./shared/artifacts/`,
asi que el contenedor de inferencia los recogera automaticamente al reiniciarse.

---

## Endpoints de la API

| Metodo | Ruta              | Descripcion                                       |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Redirige a `/docs` (Swagger UI).                  |
| GET    | `/docs`           | Documentacion interactiva (auto-generada).        |
| GET    | `/health`         | Salud del servicio (modelo cargado, umbral...).   |
| GET    | `/metadata`       | Metadatos del modelo (metricas test, fecha, ...). |
| POST   | `/predict`        | Prediccion para 1 transaccion.                    |
| POST   | `/predict/batch`  | Prediccion para hasta 10.000 transacciones.       |

Codigos de error tipicos:

- `422 Unprocessable Entity` -- el JSON no respeta el contrato (`schemas.py`).
- `503 Service Unavailable` -- los artefactos no estan cargados.
- `400 Bad Request` -- error en el `predict` (preprocesador, modelo).

---

## Variables de entorno

Copie `.env.example` a `.env` y modifique lo que necesite. Las mas utiles:

| Variable             | Default | Donde aplica   | Descripcion                                  |
|----------------------|---------|----------------|----------------------------------------------|
| `API_PORT`           | `8000`  | inference      | Puerto host -> contenedor.                   |
| `UVICORN_WORKERS`    | `1`     | inference      | Procesos uvicorn (subir si hay carga alta).  |
| `THRESHOLD_OVERRIDE` | (vacio) | inference      | Forzar un umbral fijo, ignorando metadatos.  |
| `DEFAULT_THRESHOLD`  | `0.5`   | inference      | Fallback si los metadatos no traen umbral.   |
| `SEARCH_MODE`        | `quick` | training       | `quick` o `full` (RandomizedSearchCV).       |
| `SEARCH_N_ITER`      | `30`    | training       | Iteraciones de la busqueda en modo `full`.   |
| `RANDOM_STATE`       | `42`    | training       | Semilla para reproducibilidad.               |

Reglas: cualquier ruta o parametro no hard-codeado es configurable por
variable de entorno (regla del Hito 4).

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

**Quiere parar y limpiar todo**:
```bash
docker compose down --volumes --remove-orphans
docker image rm guardianai-inference guardianai-training 2>/dev/null
```

---

## Para mas detalle

Vea [DOCUMENTACION.md](DOCUMENTACION.md) -- explicacion detallada del por
que de cada decision: division en modulos, eleccion de FastAPI, patron
del volumen compartido, modos de entrenamiento, etc.
