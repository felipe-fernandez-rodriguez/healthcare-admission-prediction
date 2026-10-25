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
- **Muestra usada:** para que cada CSV pese ~9 MB (límite de GitHub: aviso a 50 MB, bloqueo a 100 MB) se toma una **muestra aleatoria reproducible de 100 000 filas** (`seed=42`). Se ajusta en `src/config.py` (`MAX_ROWS`; `None` = dataset completo).
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
│   └── 03-model-clasification.ipynb    # EA2 clasificación + model-summary
├── reports/
│   ├── data-summary.txt                # resumen del DataFrame transformado
│   ├── model-summary.txt               # resumen de ambos modelos
│   └── *.json                          # resultados estructurados / metadata
├── src/
│   ├── config.py                       # rutas relativas y constantes
│   ├── pipeline.py                     # flujo completo por script
│   ├── data/huggingface_loader.py      # HFDatasetLoader (solo carga)
│   ├── data/preprocessing.py           # transformaciones
│   ├── models/train_regression.py      # regresión simple y múltiple
│   ├── models/train_classification.py  # clasificador de severidad
│   └── utils/reporter.py               # DatasetSummarizer y ModelSummaryReporter
├── .env.example · .gitignore · requirements.txt
```

## Instalación y ejecución local

Requisitos: Python 3.12 o similar (probado también en 3.14) y conexión a internet.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # opcional: agregue su HF_TOKEN
python -m src.pipeline             # genera datos, modelos y reportes
```

**Token de Hugging Face:** el dataset es público, así que el token es opcional. Se lee de la variable de entorno `HF_TOKEN` o del archivo `.env` (no versionado). En GitHub, agréguelo como secreto `HF_TOKEN` del repositorio (Settings → Secrets and variables → Actions).

**Notebooks** (ejecutar en orden, desde cualquier carpeta; localizan la raíz solos):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-ingestion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02-model-linear-regresion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03-model-clasification.ipynb
```

o ábralos con `jupyter lab`. El 03 necesita los resultados del 01 y el 02.

## EA1 · Carga y creación del dataset

- `HFDatasetLoader` (`src/data/huggingface_loader.py`): recibe `dataset_id`, configuración y `split`, usa `datasets.load_dataset()` y devuelve un `DataFrame`; valida identificador, configuración y split con mensajes de error claros. No transforma ni genera reportes.
- `DatasetSummarizer` (`src/utils/reporter.py`): recibe un DataFrame y escribe `reports/data-summary.txt` (filas, columnas, tipos, nulos, duplicados y estadísticas descriptivas).
- `notebooks/01-data-ingestion.ipynb`: carga, guarda `data/raw`, aplica y explica las transformaciones, guarda `data/processed` y genera el resumen **del DataFrame transformado**.

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

## Reportes y evidencias

| Archivo | Contenido |
|---|---|
| `reports/data-summary.txt` | Resumen del DataFrame transformado |
| `reports/model-summary.txt` | Dataset, preprocesamiento, ambos modelos, métricas, interpretación y limitaciones |
| `reports/*.json` | Resultados estructurados y metadata de ingestión |
| `models/*.pkl` | Pipelines entrenados (comprimidos, ~2 MB en total) |

## GitHub Actions

El workflow `.github/workflows/ml-pipeline.yml` se dispara automáticamente en cada `push` a `main` y:

1. instala Python 3.12 y `requirements.txt`;
2. ejecuta `python -m src.pipeline` (descarga el dataset desde Hugging Face);
3. ejecuta los tres notebooks con `nbconvert`;
4. falla si falta algún reporte o si algún archivo generado supera 50 MB;
5. publica `data-summary.txt` y `model-summary.txt` como artefacto **`reportes-ml`**;
6. hace commit de datos, modelos, reportes y notebooks ejecutados con `[skip ci]` (no se re-dispara).

**Dónde ver los artefactos:** pestaña **Actions** → ejecución finalizada → sección *Artifacts* → `reportes-ml`.

## Limitaciones y consideraciones éticas

- Son datos de salud: aunque el dataset es público y no contiene nombres, no deben usarse para decisiones clínicas reales. Cualquier uso real requeriría validación clínica, revisión de sesgos (edad, género, seguro) y cumplimiento de privacidad.
- Los modelos describen asociaciones, no causalidad; las métricas se calculan sobre una muestra reproducible del dataset.
- El poder predictivo de las variables disponibles es bajo (ver `model-summary.txt`).
- No suba tokens al repositorio: use `.env` (ignorado) o secretos de GitHub.

## Convención de commits sugerida

```
feat: carga dataset healthcare desde Hugging Face
feat: agrega resumen automático del dataframe
feat: implementa regresión lineal
feat: implementa clasificador de severidad
ci: automatiza pipeline de machine learning
docs: actualiza README y evidencias
```
