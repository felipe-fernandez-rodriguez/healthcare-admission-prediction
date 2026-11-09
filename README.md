# Predicción de admisiones hospitalarias (EA1 + EA2)

Proyecto de Machine Learning que carga automáticamente el dataset `Nicolybgs/healthcare_data` desde Hugging Face, lo prepara y entrena dos modelos —**regresión lineal** (depósito de admisión) y **clasificación** (severidad de la enfermedad)—, generando reportes `.txt` de forma automática con **GitHub Actions** en cada `push`.

## Integrantes

| Nombre |
|---|
| _Felipe Fernández Rodriguez_ |

## Dataset

- **Identificador:** `Nicolybgs/healthcare_data`
- **Enlace:** <https://huggingface.co/datasets/Nicolybgs/healthcare_data>
- **Configuración / split:** verificados por código con `HFDatasetLoader.available_splits()`: configuración `default`, único split `train` (500 000 filas, 15 columnas).
- **Datos completos vs. CSV guardados:** la descarga, la limpieza, el resumen y el entrenamiento de los modelos usan **las 500 000 filas**. Solo los CSV de `data/` (que se suben al repositorio) se limitan a `CSV_MAX_ROWS = 100 000` filas (muestra aleatoria reproducible, `seed=42`, ~9 MB cada uno) para no superar el peso permitido por GitHub (aviso a 50 MB, bloqueo a 100 MB). Se ajusta en `src/config.py`.
- **Columnas:** `Available Extra Rooms in Hospital`, `Department`, `Ward_Facility_Code`, `doctor_name`, `staff_available`, `patientid`, `Age` (rangos de texto), `gender`, `Type of Admission`, `Severity of Illness`, `health_conditions`, `Visitors with Patient`, `Insurance`, `Admission_Deposit` (objetivo de regresión), `Stay (in days)`.
- `Severity of Illness` ya es categórica (Minor, Moderate, Extreme), por lo que no se construyó ninguna etiqueta a partir de una variable numérica.

## Estructura

```
healthcare-admission-prediction/
├── .github/workflows/ml-pipeline.yml   # GitHub Actions (se ejecuta en cada push)
├── data/
│   ├── raw/healthcare_raw.csv          # datos sin transformar (generado)
│   └── processed/healthcare_processed.csv  # datos transformados (generado)
├── models/                             # pipelines entrenados (.pkl, generados)
├── notebooks/
│   ├── 01-data-ingestion.ipynb         # EA1
│   ├── 02-model-linear-regresion.ipynb # EA2 regresión
│   ├── 03-model-clasification.ipynb    # EA2 clasificación + model-summary
│   └── 04-model-random-forest.ipynb    # EA3 Random Forest
├── reports/
│   ├── data-summary.txt                # resumen del DataFrame transformado
│   ├── model-summary.txt               # resumen de ambos modelos
│   ├── *.json                          # resultados estructurados / metadata
│   └── random_forest/                  # resumen, findings.md, results.json, CSV y figures/
├── src/
│   ├── config.py                       # rutas relativas y constantes
│   ├── pipeline.py                     # orquestador (--stage all|ingestion|baseline-models|random-forest)
│   ├── data/huggingface_loader.py      # HFDatasetLoader (solo carga)
│   ├── data/preprocessing.py           # transformaciones
│   ├── models/train_regression.py      # regresión simple y múltiple
│   ├── models/train_classification.py  # clasificador de severidad (EA2)
│   ├── models/train_random_forest.py   # Random Forest (EA3)
│   └── utils/reporter.py               # DatasetSummarizer, ModelSummaryReporter, RandomForestReporter
├── .env.example · .gitignore · requirements.txt
```

## Instalación y ejecución local

Requisitos: Python 3.12 o similar (probado también en 3.14) y conexión a internet.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # opcional: agregue su HF_TOKEN
python -m src.pipeline             # genera datos, modelos y reportes (todas las etapas)
```

**Token de Hugging Face:** el dataset es público, así que el token es opcional. Se lee de la variable de entorno `HF_TOKEN` o del archivo `.env` (no versionado). En GitHub, agréguelo como secreto `HF_TOKEN` del repositorio (Settings → Secrets and variables → Actions).

**Notebooks** (ejecutar en orden, desde cualquier carpeta; localizan la raíz solos):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-ingestion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02-model-linear-regresion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03-model-clasification.ipynb
```

o ábralos con `jupyter lab`. El 03 necesita `reports/regression-results.json`, generado por el 02.

## EA1 · Carga y creación del dataset

- `HFDatasetLoader` (`src/data/huggingface_loader.py`): recibe `dataset_id`, configuración y `split`, usa `datasets.load_dataset()` y devuelve un `DataFrame`; valida identificador, configuración y split con mensajes de error claros. No transforma ni genera reportes.
- `DatasetSummarizer` (`src/utils/reporter.py`): recibe un DataFrame y escribe `reports/data-summary.txt` (filas, columnas, tipos, nulos, duplicados y estadísticas descriptivas).
- `notebooks/01-data-ingestion.ipynb`: descarga el dataset completo, aplica y explica las transformaciones, genera el resumen **del DataFrame transformado completo** y guarda `data/raw` y `data/processed` limitados a `CSV_MAX_ROWS` filas.
- Notebooks 02 y 03: no leen ni guardan CSV; cada uno descarga y limpia el dataset completo en memoria (`src/data/dataset_io.py`), entrena y deja sus resultados en `reports/`.

### Transformaciones y decisiones de preprocesamiento

| Transformación | Problema que resuelve |
|---|---|
| Rellenar nulos de `health_conditions` con `"None"` | El nulo significa "sin condición preexistente"; evita perder ~30 % de las filas. |
| Eliminar duplicados exactos | Evita sobreponderar registros repetidos (se hace antes de quitar `patientid`). |
| Eliminar `patientid` | Identificador sin valor predictivo; riesgo de memorización. |
| `Age` → `age_midpoint` | Convierte rangos de texto (`41-50`) en un número (45.5) usable por modelos. |
| Nombres a `snake_case` | Facilita el uso en código (`stay_in_days`). |
| Split train/test + pipelines | Imputación y codificación se ajustan solo con entrenamiento (sin fuga de información). |

## EA2 · Modelos

| | Regresión simple | Regresión múltiple | Clasificación |
|---|---|---|---|
| Objetivo | `admission_deposit` | `admission_deposit` | `severity_of_illness` (Minor/Moderate/Extreme) |
| Predictoras | `stay_in_days` | `stay_in_days`, `visitors_with_patient`, `available_extra_rooms_in_hospital`, `age_midpoint`, `insurance` (No=1, Yes=2), `severity_of_illness` (1–3) | `age_midpoint`, `health_conditions`, `type_of_admission`, `stay_in_days`, `visitors_with_patient` |
| Algoritmo | `LinearRegression` (por defecto) en `Pipeline` | `LinearRegression` (por defecto) en `Pipeline` + `ColumnTransformer` | `RandomForestClassifier` en `Pipeline` + OneHot |
| Métricas | MAE, RMSE, R² | MAE, RMSE, R² | accuracy, precisión/recall/F1 ponderados, matriz de confusión |
| División | 80/20, `random_state=42` | 80/20, `random_state=42` | 80/20 estratificada, `random_state=42` |

- Random Forest se eligió por manejar variables mixtas y relaciones no lineales sin escalado; `class_weight="balanced"` compensa el desbalance de clases. Se compara siempre contra una referencia trivial (media / clase mayoritaria).
- Los coeficientes de regresión se interpretan en unidades originales (no se escalan las numéricas).
- Los resultados reales (métricas, coeficientes, matriz de confusión e interpretación) están en `reports/model-summary.txt`; las métricas **no se escriben a mano**, salen de la ejecución. Con estos datos las relaciones son débiles (R² cercano a 0 y exactitud del clasificador por debajo de la clase mayoritaria con `balanced`); el reporte lo indica.

## EA3 · Random Forest: severidad al momento de la admisión

**Caso hipotético:** Hospital San Juan de Dios de Rionegro.

**Pregunta de negocio:** _Al momento de la admisión de un paciente, ¿con qué desempeño puede un modelo predecir su nivel de gravedad (`Severity of Illness`: Minor, Moderate, Extreme) a partir de información demográfica, antecedentes de salud, tipo de ingreso y variables logísticas disponibles, para apoyar el triaje preliminar y la planificación anticipada de camas y personal?_

**Beneficiarios y decisión apoyada:** directiva médica, coordinación de urgencias y gestión operativa. El modelo apoya, de forma **preliminar**, la planificación de camas, la distribución preventiva de personal, la anticipación de aumentos de pacientes graves y la priorización operativa inicial. No toma decisiones clínicas ni reemplaza la valoración médica.

**Pertinencia del dataset:** `Nicolybgs/healthcare_data` trae admisiones hospitalarias con la gravedad ya etiquetada (`Severity of Illness`), variables demográficas, antecedentes (`health_conditions`), tipo de ingreso y variables logísticas, con unas 500 000 filas, suficientes para entrenar y evaluar con una prueba independiente. Es un dataset público de origen no verificado, por eso se usa solo como caso de estudio.

### Variables y prevención de fuga de información

- **Predictoras (disponibles al ingreso):** `age_midpoint`, `gender`, `health_conditions`, `type_of_admission`, `visitors_with_patient`, `staff_available`, `available_extra_rooms_in_hospital`, `insurance`.
- **Excluidas:** `patientid` (identificador); `severity_of_illness` (objetivo); `stay_in_days` (se conoce después de la atención: fuga de información); `admission_deposit` (se fija durante o después de la admisión y podría reflejar la gravedad); `department`, `ward_facility_code` y `doctor_name` (se asignan tras el triaje y pueden revelar la gravedad de forma indirecta).
- Supuesto a validar con el hospital: `visitors_with_patient`, `staff_available` y `available_extra_rooms_in_hospital` se conocen en el momento de la admisión.

### Cómo funciona Random Forest (en palabras simples)

1. Se toman muchas muestras aleatorias de los pacientes de entrenamiento (con reemplazo).
2. Con cada muestra se construye un árbol de decisión; en cada división solo se consideran algunas variables elegidas al azar, para que los árboles sean distintos entre sí.
3. Cuando llega un paciente nuevo, cada árbol «vota» una severidad y gana la clase más votada.
4. Combinar muchos árboles reduce el sobreajuste de un árbol aislado.

- **Parámetros principales:** `n_estimators` (cantidad de árboles), `max_depth` (profundidad máxima), `min_samples_leaf` (mínimo de pacientes por hoja) y `class_weight` (mayor peso a clases minoritarias). Se compararon 7 configuraciones sobre una submuestra estratificada de 100 000 filas del entrenamiento. Se eligió con el F1 ponderado y el recall de la clase `Extreme` (omitir un caso grave tiene mayor costo operativo), descartando las que producirían un modelo demasiado pesado para GitHub (hojas pequeñas); el detalle está en `reports/random_forest/hyperparameter_comparison.csv`.
- **Ventajas:** captura relaciones no lineales e interacciones, maneja variables mixtas sin escalarlas y es robusto al ruido.
- **Limitaciones:** menos interpretable que un árbol único o una regresión, genera modelos más pesados y su desempeño depende de que las variables contengan señal.
- **Metodología:** división 80/20 estratificada (`random_state=42`); imputación y One-Hot se ajustan solo con entrenamiento; el modelo final se entrena con **toda** la partición de entrenamiento (400 000 filas) y se evalúa en el 20 % de prueba, independiente; importancia por permutación sobre la prueba completa; comparación con una línea base que siempre predice la clase mayoritaria.

### Cómo ejecutar EA3

```bash
python -m src.pipeline --stage random-forest   # solo el Random Forest (~1,7 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/04-model-random-forest.ipynb
```

### Dónde están los resultados

| Entregable | Ruta |
|---|---|
| Resumen técnico automático | `reports/random_forest/model-summary.txt` |
| **Documento de hallazgos para directivos** | [`reports/random_forest/findings.md`](reports/random_forest/findings.md) |
| Resultados estructurados | `reports/random_forest/results.json` |
| Comparación de hiperparámetros | `reports/random_forest/hyperparameter_comparison.csv` |
| Métricas por clase | `reports/random_forest/metrics_by_class.csv` |
| Matriz de confusión | `reports/random_forest/figures/confusion_matrix.png` |
| Importancia de variables | `reports/random_forest/figures/feature_importance.png` |
| Modelo exportado | `models/random_forest_severity.pkl` |

Las métricas y la valoración de utilidad se calculan en cada ejecución; **no** están escritas a mano. Las métricas reales de la última ejecución están en el resumen y en el documento de hallazgos.

### Consideraciones éticas y de privacidad (EA3)

Son datos de salud: el modelo es solo un apoyo administrativo, sujeto a validación clínica, revisión de sesgos (género, edad, seguro) y supervisión humana. No debe usarse para diagnosticar, decidir tratamientos ni negar atención. Un uso real exigiría datos propios del hospital, anonimización, cumplimiento de la normativa de protección de datos y monitoreo periódico del desempeño.

## Reportes y evidencias

| Archivo | Contenido |
|---|---|
| `reports/data-summary.txt` | Resumen del DataFrame transformado |
| `reports/model-summary.txt` | Dataset, preprocesamiento, ambos modelos, métricas, interpretación y limitaciones |
| `reports/*.json` | Resultados estructurados y metadata de ingestión |
| `models/*.pkl` | Pipelines entrenados (comprimidos; el Random Forest pesa ~8 MB) |

## GitHub Actions

El workflow `.github/workflows/ml-pipeline.yml` se dispara automáticamente en cada `push` y `pull_request` hacia `main` y:

1. instala Python 3.12 y `requirements.txt`;
2. ejecuta `python -m src.pipeline` (ingestión, modelos EA2 y Random Forest EA3), que descarga el dataset desde Hugging Face;
3. ejecuta los cuatro notebooks con `nbconvert`;
4. falla de forma explícita si falta algún resultado requerido o si algún archivo generado supera 50 MB;
5. publica los artefactos:
   - **`reportes-ml`**: todos los `.txt`, `.md`, `.json`, `.csv` y `.png` de `reports/` (EA1, EA2 y EA3);
   - **`modelo-random-forest`**: `models/random_forest_severity.pkl`;
   - **`notebooks-ejecutados`**: los cuatro notebooks con sus salidas.

El workflow **no hace commits automáticos**: los resultados se conservan como artefactos. Para versionar los reportes, ejecute el pipeline localmente (o descargue el artefacto) y haga commit de `reports/`.

**Dónde ver los artefactos:** pestaña **Actions** → ejecución finalizada → sección *Artifacts*.

## Limitaciones y consideraciones éticas

- Son datos de salud: aunque el dataset es público y no contiene nombres, no deben usarse para decisiones clínicas reales. Cualquier uso real requeriría validación clínica, revisión de sesgos (edad, género, seguro) y cumplimiento de privacidad.
- Los modelos describen asociaciones, no causalidad; las métricas se calculan sobre el dataset completo (partición de prueba del 20 %).
- El poder predictivo de las variables disponibles es bajo (ver `model-summary.txt`).
- No suba tokens al repositorio: use `.env` (ignorado) o secretos de GitHub.
