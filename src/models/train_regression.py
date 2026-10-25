"""Regresión lineal simple y múltiple para estimar ``admission_deposit``.

Se parte del CSV procesado (columnas en ``snake_case``). Cada modelo se entrena
dentro de un ``Pipeline`` de scikit-learn, de modo que los pasos que aprenden de
los datos (imputación, codificación) se ajustan solo con el conjunto de
entrenamiento.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from src import config

TARGET = "admission_deposit"
SIMPLE_FEATURES = ["stay_in_days"]
NUMERIC_FEATURES = [
    "stay_in_days",
    "visitors_with_patient",
    "available_extra_rooms_in_hospital",
    "age_midpoint",
]
INSURANCE_ORDER = ["No", "Yes"]  # No=1, Yes=2 tras el desplazamiento a base 1
SEVERITY_ORDER = ["Minor", "Moderate", "Extreme"]  # Minor=1, Moderate=2, Extreme=3
MULTIPLE_FEATURES = NUMERIC_FEATURES + ["insurance", "severity_of_illness"]

SIMPLE_QUESTION = (
    "¿Se puede estimar el depósito de admisión usando únicamente los días de estadía?"
)
MULTIPLE_QUESTION = (
    "¿Cómo influyen factores clínicos (severidad, edad), logísticos (habitaciones libres) "
    "y administrativos (seguro, días de estadía, visitantes) en el depósito de admisión?"
)


def _shift_to_one_based(values: np.ndarray) -> np.ndarray:
    """Convierte los códigos ordinales 0..k-1 en 1..k (Minor=1, Moderate=2, Extreme=3)."""
    return values + 1


def build_simple_pipeline() -> Pipeline:
    """Regresión lineal simple: ``stay_in_days`` -> ``admission_deposit``."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )


def build_multiple_pipeline() -> Pipeline:
    """Regresión lineal múltiple con preprocesamiento por tipo de variable.

    * Numéricas: imputación por mediana (sin escalado, para conservar la
      interpretación de los coeficientes en las unidades originales).
    * ``insurance``: No=1, Yes=2.
    * ``severity_of_illness``: Minor=1, Moderate=2, Extreme=3.
    """

    def ordinal(order: list[str]) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[order])),
                ("shift", FunctionTransformer(_shift_to_one_based, feature_names_out="one-to-one")),
            ]
        )

    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("insurance", ordinal(INSURANCE_ORDER), ["insurance"]),
            ("severity", ordinal(SEVERITY_ORDER), ["severity_of_illness"]),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


@dataclass
class RegressionRun:
    """Resultado de una ejecución: pipeline entrenado, datos de prueba y resumen."""

    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    results: dict[str, Any]


def _interpret_coefficients(
    coefficients: dict[str, float], intercept: float, simple: bool
) -> list[str]:
    notes = [
        f"Intercepto: con todas las variables predictoras en 0 el depósito estimado sería "
        f"{intercept:,.2f} (referencia matemática; puede no ser un caso realista)."
    ]
    descriptions = {
        "stay_in_days": "cada día adicional de estadía",
        "visitors_with_patient": "cada visitante adicional",
        "available_extra_rooms_in_hospital": "cada habitación libre adicional",
        "age_midpoint": "cada año adicional de edad (punto medio del rango)",
        "insurance": "pasar de insurance=No (1) a Yes (2)",
        "severity_of_illness": "subir un nivel de severidad (Minor=1, Moderate=2, Extreme=3)",
    }
    suffix = "" if simple else ", manteniendo constantes las demás variables"
    for name, coef in coefficients.items():
        direction = "aumenta" if coef > 0 else "reduce"
        notes.append(
            f"{name}: {descriptions.get(name, name)} {direction} el depósito estimado en "
            f"{abs(coef):,.2f} unidades{suffix}."
        )
    return notes


def _limitations(r2: float) -> list[str]:
    explained = max(r2, 0.0) * 100
    return [
        f"El modelo explica aproximadamente {explained:.1f} % de la varianza del depósito en prueba (R²={r2:.4f}).",
        "La regresión lineal asume relaciones lineales y aditivas; no captura interacciones ni no linealidades.",
        "Los coeficientes describen asociaciones, no causalidad.",
        "Las categorías ordinales (severidad) asumen separaciones equidistantes entre niveles.",
        "Los resultados provienen de una muestra reproducible del dataset, no de todos sus registros.",
    ]


def evaluate_regression(
    df: pd.DataFrame,
    *,
    name: str,
    question: str,
    pipeline: Pipeline,
    features: list[str],
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
    model_path: str | Path | None = None,
) -> RegressionRun:
    """Entrena y evalúa un pipeline de regresión; devuelve un resumen serializable."""
    missing = [c for c in [TARGET, *features] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en el DataFrame: {missing}")

    data = df.dropna(subset=[TARGET])
    X, y = data[features], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    baseline_mae = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))

    model: LinearRegression = pipeline.named_steps["model"]
    if "preprocess" in pipeline.named_steps:
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    else:
        names = list(features)
    coefficients = {n: float(c) for n, c in zip(names, model.coef_)}
    intercept = float(model.intercept_)
    simple = len(features) == 1

    saved = None
    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path, compress=3)
        saved = str(path.relative_to(config.PROJECT_ROOT)) if path.is_relative_to(config.PROJECT_ROOT) else str(path)

    results: dict[str, Any] = {
        "name": name,
        "question": question,
        "target": TARGET,
        "predictors": list(features),
        "algorithm": "LinearRegression (mínimos cuadrados ordinarios) dentro de un Pipeline",
        "parameters": {
            "LinearRegression": {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in model.get_params().items()
            },
            "nota": "Se usan los valores predeterminados de LinearRegression.",
            "pasos_pipeline": [step for step, _ in pipeline.steps],
        },
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAE_baseline_media": baseline_mae,
        },
        "intercept": intercept,
        "coefficients": coefficients,
        "interpretation": [
            f"MAE={mae:,.2f}: error absoluto medio de la predicción del depósito en el conjunto de prueba.",
            f"RMSE={rmse:,.2f}: error cuadrático medio (penaliza más los errores grandes).",
            f"R²={r2:.4f}: proporción de la varianza del depósito explicada por el modelo.",
            f"Referencia: predecir siempre la media de entrenamiento da MAE={baseline_mae:,.2f}; "
            + (
                "el modelo no mejora esa referencia."
                if mae >= baseline_mae
                else "el modelo mejora esa referencia solo de forma marginal (<2 %)."
                if (baseline_mae - mae) / baseline_mae < 0.02
                else "el modelo mejora esa referencia."
            ),
            *_interpret_coefficients(coefficients, intercept, simple),
        ],
        "limitations": _limitations(r2),
        "model_path": saved,
    }
    return RegressionRun(pipeline, X_test, y_test, y_pred, results)


def run_simple_regression(df: pd.DataFrame, save_model: bool = True) -> RegressionRun:
    """Regresión lineal simple: depósito vs. días de estadía."""
    return evaluate_regression(
        df,
        name="Regresión lineal simple",
        question=SIMPLE_QUESTION,
        pipeline=build_simple_pipeline(),
        features=SIMPLE_FEATURES,
        model_path=config.MODELS_DIR / "regression_simple.pkl" if save_model else None,
    )


def run_multiple_regression(df: pd.DataFrame, save_model: bool = True) -> RegressionRun:
    """Regresión lineal múltiple con variables numéricas y categóricas codificadas."""
    return evaluate_regression(
        df,
        name="Regresión lineal múltiple",
        question=MULTIPLE_QUESTION,
        pipeline=build_multiple_pipeline(),
        features=MULTIPLE_FEATURES,
        model_path=config.MODELS_DIR / "regression_multiple.pkl" if save_model else None,
    )


def run_regression(df: pd.DataFrame, save_model: bool = True) -> dict[str, Any]:
    """Ejecuta ambos modelos y devuelve ``{"simple": {...}, "multiple": {...}}``."""
    return {
        "simple": run_simple_regression(df, save_model).results,
        "multiple": run_multiple_regression(df, save_model).results,
    }
