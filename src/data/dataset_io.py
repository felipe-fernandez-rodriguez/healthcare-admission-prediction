"""Flujo de datos compartido: descarga completa, limpieza y guardado limitado en CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.data.huggingface_loader import HFDatasetLoader
from src.data.preprocessing import PREPROCESSING_DESCRIPTION, preprocess


def make_loader() -> HFDatasetLoader:
    """Loader del dataset del proyecto (siempre completo, sin muestreo)."""
    return HFDatasetLoader(
        config.DATASET_ID,
        config=config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
    )


def download_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Descarga el dataset completo desde Hugging Face y lo limpia.

    No escribe archivos: es el punto de entrada de los notebooks 02 y 03.

    Returns:
        ``(raw, processed, info)`` con el DataFrame original, el limpio y la
        información de la carga.
    """
    loader = make_loader()
    raw = loader.load()
    return raw, preprocess(raw), loader.info()


def save_csv_capped(
    df: pd.DataFrame,
    path: str | Path,
    max_rows: int | None = config.CSV_MAX_ROWS,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Guarda ``df`` como CSV limitando las filas (muestra reproducible).

    Si ``df`` tiene más de ``max_rows`` filas se guarda una muestra aleatoria
    con semilla fija; si no, se guarda completo. El DataFrame original no se modifica.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = df.sample(n=max_rows, random_state=seed) if max_rows and len(df) > max_rows else df
    out.to_csv(target, index=False)
    return {
        "archivo": target.relative_to(config.PROJECT_ROOT).as_posix()
        if target.is_relative_to(config.PROJECT_ROOT)
        else str(target),
        "filas_guardadas": len(out),
        "filas_totales": len(df),
        "tamano_mb": round(target.stat().st_size / 1e6, 2),
    }


def build_metadata(
    info: dict[str, Any],
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    saved_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Metadatos de la ingestión que consume ``ModelSummaryReporter``."""
    return {
        "dataset_id": config.DATASET_ID,
        "dataset_url": config.DATASET_URL,
        "config": info["config"],
        "split": info["split"],
        "filas_originales_en_hf": info["filas_originales_en_hf"],
        "filas_cargadas": len(raw),
        "muestreo": info["muestreo"],
        "columnas_iniciales": list(raw.columns),
        "filas_procesadas": len(processed),
        "columnas_procesadas": list(processed.columns),
        "preprocesamiento": PREPROCESSING_DESCRIPTION,
        "csv_guardados": saved_files or [],
    }
