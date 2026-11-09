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


def build_preprocessing_evidence(
    raw: pd.DataFrame, processed: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Tabla de evidencia antes/después del preprocesamiento.

    Una fila por métrica con su valor en el DataFrame original, en el procesado y,
    si se indican ``features``, en la matriz de predictoras final.
    """
    rows: list[dict[str, Any]] = [
        {"metrica": "filas", "antes": len(raw), "despues": len(processed)},
        {"metrica": "columnas", "antes": raw.shape[1], "despues": processed.shape[1]},
        {"metrica": "filas duplicadas", "antes": int(raw.duplicated().sum()), "despues": int(processed.duplicated().sum())},
        {"metrica": "valores nulos (total)", "antes": int(raw.isna().sum().sum()), "despues": int(processed.isna().sum().sum())},
        {"metrica": "columnas de texto (object)", "antes": int((raw.dtypes == "object").sum()), "despues": int((processed.dtypes == "object").sum())},
        {"metrica": "columnas numéricas", "antes": int(raw.select_dtypes("number").shape[1]), "despues": int(processed.select_dtypes("number").shape[1])},
        {"metrica": "columnas eliminadas", "antes": "", "despues": ", ".join(sorted(set(raw.columns) - set(processed.columns)))},
        {"metrica": "columnas creadas/renombradas", "antes": "", "despues": ", ".join(sorted(set(processed.columns) - set(raw.columns)))},
    ]
    for col in raw.columns[raw.isna().any()]:
        rows.append({"metrica": f"nulos en {col}", "antes": int(raw[col].isna().sum()), "despues": ""})
    if features is not None:
        rows.append({"metrica": "variables predictoras finales", "antes": "", "despues": ", ".join(features)})
    return pd.DataFrame(rows)
