"""Pipeline completo: Hugging Face -> CSV raw/processed -> modelos -> reportes.

Uso (desde la raíz del proyecto)::

    python -m src.pipeline
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.data.dataset_io import build_metadata, download_and_clean, save_csv_capped
from src.models.train_classification import run_classification
from src.models.train_regression import run_regression
from src.utils.reporter import (
    DatasetSummarizer,
    ModelSummaryReporter,
    save_json,
)


def run_ingestion() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """EA1: descarga completa, limpieza, CSV limitados y resumen del DataFrame completo."""
    raw, processed, info = download_and_clean()
    saved = [
        save_csv_capped(raw, config.RAW_CSV),
        save_csv_capped(processed, config.PROCESSED_CSV),
    ]
    DatasetSummarizer().save(processed, config.DATA_SUMMARY_TXT)
    metadata = build_metadata(info, raw, processed, saved)
    save_json(metadata, config.INGESTION_METADATA_JSON)
    return raw, processed, metadata


def main() -> None:
    """Ejecuta ingestión, regresión, clasificación y los dos reportes.

    Los modelos se entrenan con el DataFrame completo, no con los CSV guardados.
    """
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
