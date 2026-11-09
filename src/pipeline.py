"""Pipeline completo: Hugging Face -> CSV raw/processed -> modelos -> reportes.

Uso (desde la raíz del proyecto)::

    python -m src.pipeline                          # todas las etapas
    python -m src.pipeline --stage ingestion        # solo ingestión y resumen de datos
    python -m src.pipeline --stage baseline-models  # regresión lineal + clasificador base
    python -m src.pipeline --stage random-forest    # Random Forest de severidad
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.data.dataset_io import (
    build_metadata,
    build_preprocessing_evidence,
    download_and_clean,
    save_csv_capped,
)
from src.models.train_classification import run_classification
from src.models.train_random_forest import (
    FEATURES,
    RandomForestRun,
    export_model,
    train_random_forest,
)
from src.models.train_regression import run_regression
from src.utils.reporter import (
    DatasetSummarizer,
    ModelSummaryReporter,
    RandomForestReporter,
    save_json,
)

STAGES = ("all", "ingestion", "baseline-models", "random-forest")


def run_ingestion(
    raw: pd.DataFrame, processed: pd.DataFrame, info: dict[str, Any]
) -> dict[str, Any]:
    """Guarda los CSV (limitados por peso), el resumen del DataFrame y la metadata."""
    saved = [
        save_csv_capped(raw, config.RAW_CSV),
        save_csv_capped(processed, config.PROCESSED_CSV),
    ]
    DatasetSummarizer().save(processed, config.DATA_SUMMARY_TXT)
    metadata = build_metadata(info, raw, processed, saved)
    save_json(metadata, config.INGESTION_METADATA_JSON)
    return metadata


def run_baseline_models(processed: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    """Regresión lineal y clasificador base; genera ``reports/model-summary.txt``."""
    regression = run_regression(processed)
    save_json(regression, config.REGRESSION_RESULTS_JSON)
    classification = run_classification(processed).results
    save_json(classification, config.CLASSIFICATION_RESULTS_JSON)
    return ModelSummaryReporter().save(metadata, regression, classification)


def run_random_forest(
    raw: pd.DataFrame, processed: pd.DataFrame, info: dict[str, Any]
) -> tuple[RandomForestRun, dict[str, Path]]:
    """Entrena el Random Forest con todo el entrenamiento, lo exporta y genera sus reportes."""
    metadata = build_metadata(info, raw, processed)
    evidence = build_preprocessing_evidence(raw, processed, FEATURES)
    run = train_random_forest(processed)
    export_model(run)
    paths = RandomForestReporter(run, metadata, evidence).save_all()
    paths["model"] = Path(run.results["model_path"])
    return run, paths


def main(stage: str = "all") -> None:
    """Descarga el dataset completo una sola vez y ejecuta la(s) etapa(s) pedida(s).

    Los modelos se entrenan con el DataFrame completo, no con los CSV guardados.
    """
    if stage not in STAGES:
        raise ValueError(f"Etapa inválida {stage!r}. Opciones: {STAGES}")
    raw, processed, info = download_and_clean()
    metadata = build_metadata(info, raw, processed)

    if stage in ("all", "ingestion"):
        metadata = run_ingestion(raw, processed, info)
        print(f"data-summary.txt -> {config.DATA_SUMMARY_TXT}")
    if stage in ("all", "baseline-models"):
        print(f"model-summary.txt -> {run_baseline_models(processed, metadata)}")
    if stage in ("all", "random-forest"):
        run, paths = run_random_forest(raw, processed, info)
        for name, path in paths.items():
            print(f"{name}: {path}")
        m = run.results["metrics"]
        print(
            f"accuracy={m['accuracy']:.4f} f1_weighted={m['f1_weighted']:.4f} "
            f"recall_extreme={run.results['recall_extreme']:.4f} "
            f"-> {run.results['utility_assessment']['level']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="all")
    main(parser.parse_args().stage)
