"""Configuración central del proyecto: rutas relativas a la raíz y constantes."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

RAW_CSV: Path = DATA_RAW_DIR / "healthcare_raw.csv"
PROCESSED_CSV: Path = DATA_PROCESSED_DIR / "healthcare_processed.csv"
DATA_SUMMARY_TXT: Path = REPORTS_DIR / "data-summary.txt"
MODEL_SUMMARY_TXT: Path = REPORTS_DIR / "model-summary.txt"
INGESTION_METADATA_JSON: Path = REPORTS_DIR / "ingestion-metadata.json"
REGRESSION_RESULTS_JSON: Path = REPORTS_DIR / "regression-results.json"
CLASSIFICATION_RESULTS_JSON: Path = REPORTS_DIR / "classification-results.json"

DATASET_ID: str = "Nicolybgs/healthcare_data"
DATASET_URL: str = "https://huggingface.co/datasets/Nicolybgs/healthcare_data"
DATASET_CONFIG: str | None = None  # el dataset solo expone la configuración "default"
DATASET_SPLIT: str = "train"

# Los modelos y el resumen usan SIEMPRE el dataset completo. Solo los CSV que se
# guardan en el repositorio se limitan a esta cantidad de filas (muestra aleatoria
# reproducible) para no superar el peso permitido por GitHub (aviso a 50 MB,
# bloqueo a 100 MB por archivo). ``None`` guarda todo.
CSV_MAX_ROWS: int | None = 100_000
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2

# ---- Random Forest (severidad al momento de la admisión) ----
RF_DIR: Path = REPORTS_DIR / "random_forest"
RF_FIGURES_DIR: Path = RF_DIR / "figures"
RF_SUMMARY_TXT: Path = RF_DIR / "model-summary.txt"
RF_FINDINGS_MD: Path = RF_DIR / "findings.md"
RF_RESULTS_JSON: Path = RF_DIR / "results.json"
RF_HYPERPARAMETERS_CSV: Path = RF_DIR / "hyperparameter_comparison.csv"
RF_METRICS_BY_CLASS_CSV: Path = RF_DIR / "metrics_by_class.csv"
RF_CONFUSION_PNG: Path = RF_FIGURES_DIR / "confusion_matrix.png"
RF_IMPORTANCE_PNG: Path = RF_FIGURES_DIR / "feature_importance.png"
RF_MODEL_PKL: Path = MODELS_DIR / "random_forest_severity.pkl"

# El modelo final se entrena con TODA la partición de entrenamiento. Solo la búsqueda
# de hiperparámetros usa una submuestra estratificada (para acotar el tiempo de CI).
RF_SEARCH_ROWS: int = 100_000
# Tamaño máximo estimado del modelo exportado (GitHub bloquea archivos de más de 100 MB).
RF_MAX_MODEL_MB: float = 40.0
