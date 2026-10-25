"""Clasificación multiclase de ``severity_of_illness`` (triaje predictivo)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

TARGET = "severity_of_illness"
CLASS_ORDER = ["Minor", "Moderate", "Extreme"]
NUMERIC_FEATURES = ["age_midpoint", "stay_in_days", "visitors_with_patient"]
CATEGORICAL_FEATURES = ["health_conditions", "type_of_admission"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

QUESTION = (
    "¿Se puede anticipar el nivel de gravedad de un paciente a partir de su edad, "
    "condiciones preexistentes, tipo de ingreso, días de estadía y visitantes?"
)
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": config.RANDOM_SEED,
}


def build_classification_pipeline() -> Pipeline:
    """Preprocesamiento (imputación + OneHot) y ``RandomForestClassifier``.

    Random Forest maneja relaciones no lineales e interacciones entre variables
    mixtas sin necesidad de escalado, y ``class_weight="balanced"`` compensa el
    desbalance entre clases. ``max_depth`` y ``min_samples_leaf`` limitan el
    sobreajuste y el tamaño del modelo guardado.
    """
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", RandomForestClassifier(**RF_PARAMS))]
    )


@dataclass
class ClassificationRun:
    """Resultado de una ejecución: pipeline entrenado, datos de prueba y resumen."""

    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    results: dict[str, Any]


def _interpret(
    labels: list[str], cm: np.ndarray, per_class: dict[str, dict[str, float]],
    accuracy: float, baseline_acc: float,
) -> list[str]:
    best = max(labels, key=lambda c: per_class[c]["recall"])
    worst = min(labels, key=lambda c: per_class[c]["recall"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    verdict = (
        "La exactitud queda por debajo de esa referencia: class_weight='balanced' sacrifica "
        "exactitud global para mejorar el recall de las clases minoritarias."
        if accuracy < baseline_acc
        else "El modelo supera la referencia de la clase mayoritaria."
    )
    return [
        f"Exactitud={accuracy:.4f} frente a {baseline_acc:.4f} de predecir siempre la clase mayoritaria. {verdict}",
        f"Clase mejor reconocida (mayor recall): {best} (recall={per_class[best]['recall']:.3f}).",
        f"Clase peor reconocida (menor recall): {worst} (recall={per_class[worst]['recall']:.3f}).",
        f"Error más frecuente: pacientes {labels[i]} clasificados como {labels[j]} ({int(off[i, j])} casos).",
    ]


def evaluate_classification(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
    model_path: str | Path | None = None,
) -> ClassificationRun:
    """Entrena y evalúa el clasificador con división estratificada."""
    missing = [c for c in [TARGET, *FEATURES] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en el DataFrame: {missing}")

    data = df.dropna(subset=[TARGET])
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = [c for c in CLASS_ORDER if c in set(y)] + sorted(set(y) - set(CLASS_ORDER))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
        for c in labels
    }
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = float(accuracy_score(y_test, baseline.predict(X_test)))

    model: RandomForestClassifier = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    top = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])[:8]

    saved = None
    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path, compress=3)
        saved = str(path.relative_to(config.PROJECT_ROOT)) if path.is_relative_to(config.PROJECT_ROOT) else str(path)

    results: dict[str, Any] = {
        "name": "Clasificación de severidad",
        "question": QUESTION,
        "target": TARGET,
        "classes": labels,
        "predictors": FEATURES,
        "algorithm": "RandomForestClassifier dentro de un Pipeline (imputación + OneHotEncoder)",
        "parameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in RF_PARAMS.items()},
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratified": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "class_distribution_test": {c: int((y_test == c).sum()) for c in labels},
        "metrics": {
            "accuracy": accuracy,
            "precision_weighted": float(p),
            "recall_weighted": float(r),
            "f1_weighted": float(f1),
            "accuracy_baseline_mayoritaria": baseline_acc,
        },
        "metrics_justification": (
            "Las clases están desbalanceadas, por eso además de la exactitud se reportan "
            "precisión, recall y F1 ponderados por soporte y el desglose por clase."
        ),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "top_feature_importances": [{"feature": str(n), "importance": float(v)} for n, v in top],
        "interpretation": _interpret(labels, cm, per_class, accuracy, baseline_acc),
        "limitations": [
            "Las variables predictoras disponibles tienen poder discriminante limitado; "
            "las métricas muestran cuánto, y no deben usarse como herramienta clínica real.",
            "class_weight='balanced' mejora el recall de las clases minoritarias a costa de precisión.",
            "Edad y días de estadía pueden ser consecuencia de la gravedad, no solo causa predictiva.",
            "Datos de salud: cualquier uso real exigiría validación clínica, auditoría de sesgos y privacidad.",
        ],
        "model_path": saved,
    }
    return ClassificationRun(pipeline, X_test, y_test, y_pred, results)


def run_classification(df: pd.DataFrame, save_model: bool = True) -> ClassificationRun:
    """Ejecuta el clasificador y guarda el pipeline en ``models/``."""
    return evaluate_classification(
        df, model_path=config.MODELS_DIR / "classifier_severity.pkl" if save_model else None
    )
