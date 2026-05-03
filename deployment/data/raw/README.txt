Coloque aqui Base.csv (Bank Account Fraud Suite, NeurIPS 2022) si desea
re-entrenar el modelo desde cero con el contenedor de entrenamiento.

Descarga oficial:
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

El contenedor de entrenamiento espera la ruta absoluta /data/raw/Base.csv
(equivalente a este fichero, montado en read-only por docker-compose).

NOTA: si solo desea servir inferencias con el modelo ya entrenado del Hito 3,
no necesita descargar Base.csv. Ejecute simplemente:

    docker compose up

y el contenedor 'bootstrap' copiara los artefactos preentrenados desde
../models/ a ./shared/artifacts/.
