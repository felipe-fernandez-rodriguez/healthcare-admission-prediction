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
