"""Pipeline completo: Hugging Face -> CSV raw/processed -> modelos -> reportes.

Uso (desde la raíz del proyecto)::

    python -m src.pipeline
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.data.huggingface_loader import HFDatasetLoader
from src.data.preprocessing import PREPROCESSING_DESCRIPTION, preprocess
from src.models.train_classification import run_classification
from src.models.train_regression import run_regression
from src.utils.reporter import (
    DatasetSummarizer,
    ModelSummaryReporter,
    save_json,
)


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """EA1: descarga, guarda raw, preprocesa, guarda processed y resume."""
    loader = HFDatasetLoader(
        config.DATASET_ID,
        config=config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
        max_rows=config.MAX_ROWS,
        seed=config.RANDOM_SEED,
    )
    raw = loader.load()
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.RAW_CSV, index=False)

    processed = preprocess(raw)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(config.PROCESSED_CSV, index=False)
    DatasetSummarizer().save(processed, config.DATA_SUMMARY_TXT)

    info = loader.info()
    metadata = {
        "dataset_id": config.DATASET_ID,
        "dataset_url": config.DATASET_URL,
        "config": info["config"],
        "split": info["split"],
        "filas_originales_en_hf": info["filas_originales_en_hf"],
        "filas_cargadas": info["filas_cargadas"],
        "muestreo": info["muestreo"],
        "columnas_iniciales": info["columnas"],
        "filas_procesadas": len(processed),
        "columnas_procesadas": list(processed.columns),
        "preprocesamiento": PREPROCESSING_DESCRIPTION,
    }
    save_json(metadata, config.INGESTION_METADATA_JSON)
    return raw, processed, metadata


def main() -> None:
    """Ejecuta ingestión, regresión, clasificación y los dos reportes."""
    _, processed, metadata = run_ingestion()
    print(f"data-summary.txt -> {config.DATA_SUMMARY_TXT}")

    regression = run_regression(processed)
    save_json(regression, config.REGRESSION_RESULTS_JSON)
    classification = run_classification(processed).results
    save_json(classification, config.CLASSIFICATION_RESULTS_JSON)

    path = ModelSummaryReporter().save(metadata, regression, classification)
    print(f"model-summary.txt -> {path}")


if __name__ == "__main__":
    main()
