"""Transformaciones de limpieza del dataset de admisiones hospitalarias.

Cada paso es una función pura (devuelve un DataFrame nuevo) para poder
mostrarlo por separado en el notebook y reutilizarlo en el pipeline.
"""
from __future__ import annotations

import re

import pandas as pd

ID_COLUMN = "patientid"
AGE_COLUMN = "Age"
NULL_CATEGORY_COLUMN = "health_conditions"
NO_CONDITION_LABEL = "None"


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas exactas (evita sobreponderar registros repetidos)."""
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza nulos de ``health_conditions`` por la categoría ``"None"``.

    En este dataset el valor ausente representa "sin condición preexistente",
    no un dato perdido; se hace explícito para no descartar ~30 % de filas.
    """
    out = df.copy()
    out[NULL_CATEGORY_COLUMN] = out[NULL_CATEGORY_COLUMN].fillna(NO_CONDITION_LABEL)
    return out


def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina ``patientid``: identificador sin valor predictivo (riesgo de fuga)."""
    return df.drop(columns=[ID_COLUMN])


def age_range_to_midpoint(value: str) -> float:
    """Convierte un rango como ``"41-50"`` en su punto medio (45.5)."""
    low, high = (int(part) for part in str(value).split("-"))
    return (low + high) / 2


def encode_age_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza ``Age`` (rangos de texto) por ``age_midpoint`` numérico."""
    out = df.copy()
    out["age_midpoint"] = out[AGE_COLUMN].map(age_range_to_midpoint).astype("float64")
    return out.drop(columns=[AGE_COLUMN])


def to_snake_case(name: str) -> str:
    """``"Stay (in days)"`` -> ``"stay_in_days"``."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.strip("_").lower()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra todas las columnas a ``snake_case``."""
    return df.rename(columns={c: to_snake_case(c) for c in df.columns})


def strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes en columnas de texto."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip()
    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las transformaciones en orden y devuelve el DataFrame analítico."""
    steps = (
        strip_text_columns,
        drop_duplicates,
        fill_missing_conditions,
        drop_identifier,
        encode_age_midpoint,
        normalize_column_names,
    )
    for step in steps:
        df = step(df)
    return df


PREPROCESSING_DESCRIPTION: list[str] = [
    "Se eliminan espacios sobrantes en columnas de texto.",
    "Se eliminan filas duplicadas exactas.",
    "Los nulos de health_conditions se reemplazan por la categoría 'None' (sin condición preexistente).",
    "Se elimina patientid por ser un identificador sin valor predictivo.",
    "Age (rangos de texto) se convierte a age_midpoint, el punto medio numérico del rango.",
    "Los nombres de columnas se normalizan a snake_case.",
    "En los modelos: separación train/test con semilla fija; imputación y codificación "
    "(Ordinal/OneHot) se ajustan únicamente con el conjunto de entrenamiento dentro de un Pipeline.",
]
