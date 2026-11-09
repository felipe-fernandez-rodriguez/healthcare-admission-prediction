"""Random Forest para predecir la severidad al momento de la admisión.

Caso hipotético: Hospital San Juan de Dios de Rionegro. El modelo es un apoyo
administrativo y preliminar (camas, personal, priorización operativa); no es una
herramienta de diagnóstico clínico.

Flujo: validar columnas -> dividir train/test (estratificado) -> comparar
hiperparámetros en una submuestra de entrenamiento -> seleccionar -> entrenar el
modelo final -> evaluar en prueba -> importancia por permutación.
"""
from __future__ import annotations

import io
import time
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

TARGET = "severity_of_illness"
CLASS_ORDER = ["Minor", "Moderate", "Extreme"]
CRITICAL_CLASS = "Extreme"

NUMERIC_FEATURES = [
    "age_midpoint",
    "visitors_with_patient",
    "staff_available",
    "available_extra_rooms_in_hospital",
]
CATEGORICAL_FEATURES = ["gender", "health_conditions", "type_of_admission", "insurance"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columnas del DataFrame procesado que NO se usan como predictoras y por qué.
EXCLUDED_COLUMNS: dict[str, str] = {
    "severity_of_illness": "Es la variable objetivo.",
    "stay_in_days": "Se conoce después de la atención; usarla sería fuga de información (data leakage).",
    "admission_deposit": "Se calcula/cobra durante o después de la admisión y podría reflejar la gravedad; "
    "por seguridad metodológica se excluye.",
    "department": "Se asigna tras el triaje y puede revelar indirectamente la gravedad.",
    "ward_facility_code": "Se asigna después de valorar al paciente; puede revelar la gravedad.",
    "doctor_name": "Depende de la asignación posterior al triaje; puede revelar la gravedad.",
    "patientid": "Identificador sin valor predictivo (ya eliminado en el preprocesamiento).",
}

# Configuraciones razonables a comparar (ver `compare_hyperparameters`).
CANDIDATE_CONFIGS: list[dict[str, Any]] = [
    {"name": "RF-1", "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 100, "class_weight": None},
    {"name": "RF-2", "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 100, "class_weight": "balanced_subsample"},
    {"name": "RF-3", "n_estimators": 200, "max_depth": 15, "min_samples_leaf": 100, "class_weight": "balanced_subsample"},
    {"name": "RF-4", "n_estimators": 200, "max_depth": 15, "min_samples_leaf": 200, "class_weight": "balanced_subsample"},
    {"name": "RF-5", "n_estimators": 200, "max_depth": 25, "min_samples_leaf": 50, "class_weight": "balanced_subsample"},
    {"name": "RF-6", "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5, "class_weight": None},
    {"name": "RF-7", "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "class_weight": None},
]
# Con el dataset completo, hojas pequeñas (RF-6, RF-7) producen modelos de cientos de MB: se
# evalúan como referencia, pero quedan fuera de la selección por no ser exportables a GitHub.

# Criterio interno (documentado) para calificar la utilidad operativa del modelo.
USEFUL_F1_LIFT = 0.15
USEFUL_EXTREME_RECALL = 0.60
LIMITED_F1_LIFT = 0.05


def build_pipeline(params: dict[str, Any], random_state: int = config.RANDOM_SEED) -> Pipeline:
    """Pipeline de imputación + OneHot + ``RandomForestClassifier``.

    No se escalan las variables: los árboles dividen por umbrales y no dependen
    de distancias, así que la escala no cambia el resultado.
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
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n: int, random_state: int = config.RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Submuestra estratificada reproducible (devuelve todo si ``len(X) <= n``)."""
    if len(X) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_state)
    return X_s, y_s


def _model_size_mb(pipeline: Pipeline) -> float:
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer, compress=3)
    return buffer.getbuffer().nbytes / 1e6


def _class_metrics(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        c: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])}
        for i, c in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "per_class": per_class,
    }


def compare_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configs: list[dict[str, Any]] = CANDIDATE_CONFIGS,
    search_rows: int = config.RF_SEARCH_ROWS,
    final_rows: int | None = None,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Compara configuraciones en una submuestra de entrenamiento (sin tocar prueba).

    La submuestra estratificada de ``search_rows`` filas se divide 75/25 en ajuste
    y validación. Se mide F1 ponderado, recall de la clase crítica, tiempo de
    ajuste y tamaño estimado del modelo final (extrapolado a ``final_rows``).
    """
    final_rows = final_rows or len(X_train)
    X_s, y_s = stratified_sample(X_train, y_train, search_rows, random_state)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_s, y_s, test_size=0.25, stratify=y_s, random_state=random_state
    )
    labels = [c for c in CLASS_ORDER if c in set(y_s)]
    rows = []
    for cfg in configs:
        pipe = build_pipeline(cfg, random_state)
        start = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - start
        m = _class_metrics(y_val, pipe.predict(X_val), labels)
        rows.append(
            {
                "config": cfg["name"],
                "n_estimators": cfg["n_estimators"],
                "max_depth": "None" if cfg["max_depth"] is None else cfg["max_depth"],
                "min_samples_leaf": cfg["min_samples_leaf"],
                "class_weight": "None" if cfg["class_weight"] is None else cfg["class_weight"],
                "accuracy": m["accuracy"],
                "f1_weighted": m["f1_weighted"],
                "recall_extreme": m["per_class"].get(CRITICAL_CLASS, {}).get("recall", float("nan")),
                "fit_seconds": fit_seconds,
                "est_final_model_mb": _model_size_mb(pipe) * final_rows / len(X_fit),
            }
        )
    return pd.DataFrame(rows).round(4)


def select_configuration(
    comparison: pd.DataFrame,
    tolerance: float = 0.01,
    max_model_mb: float = config.RF_MAX_MODEL_MB,
) -> tuple[dict[str, Any], str]:
    """Selecciona la configuración final.

    Criterio: entre las configuraciones cuyo modelo exportado estimado no supera
    ``max_model_mb``, se toman las que están a ``tolerance`` del mejor F1
    ponderado y, de ellas, se elige la de mayor recall de la clase ``Extreme``
    (omitir un paciente potencialmente grave tiene mayor costo operativo). El
    desempate es el menor tiempo de ajuste.
    """
    eligible = comparison[comparison["est_final_model_mb"] <= max_model_mb]
    if eligible.empty:
        raise RuntimeError(
            f"Ninguna configuración cabe en {max_model_mb} MB; reduzca max_depth o aumente min_samples_leaf."
        )
    best_f1 = eligible["f1_weighted"].max()
    shortlist = eligible[eligible["f1_weighted"] >= best_f1 - tolerance]
    chosen = shortlist.sort_values(["recall_extreme", "fit_seconds"], ascending=[False, True]).iloc[0]
    params = next(c for c in CANDIDATE_CONFIGS if c["name"] == chosen["config"])
    rationale = (
        f"Se eligió {chosen['config']}: entre las configuraciones con F1 ponderado a menos de "
        f"{tolerance} del mejor ({best_f1:.4f}) y modelo exportado estimado <= {max_model_mb:.0f} MB, "
        f"es la de mayor recall de la clase {CRITICAL_CLASS} ({chosen['recall_extreme']:.4f}). "
        f"Tiempo de ajuste en la submuestra: {chosen['fit_seconds']:.1f} s."
    )
    return dict(params), rationale


def _group_internal_importance(pipeline: Pipeline) -> pd.Series:
    """Suma la importancia interna de las columnas One-Hot por variable original."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    grouped: dict[str, float] = {}
    for name, value in zip(names, values):
        original = name
        for col in CATEGORICAL_FEATURES:
            if name.startswith(f"cat__{col}_"):
                original = col
                break
        else:
            original = name.removeprefix("num__")
        grouped[original] = grouped.get(original, 0.0) + float(value)
    return pd.Series(grouped).sort_values(ascending=False)


def _assess_utility(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    lift = metrics["f1_weighted"] - baseline["f1_weighted"]
    recall_extreme = metrics["per_class"][CRITICAL_CLASS]["recall"]
    if lift >= USEFUL_F1_LIFT and recall_extreme >= USEFUL_EXTREME_RECALL:
        level = "apto para apoyo operativo preliminar"
    elif lift >= LIMITED_F1_LIFT:
        level = "utilidad limitada"
    else:
        level = "no aporta valor operativo frente a la línea base"
    return {
        "level": level,
        "f1_lift_vs_baseline": float(lift),
        "recall_extreme": float(recall_extreme),
        "criteria": (
            f"Apto si el F1 ponderado supera a la línea base en >= {USEFUL_F1_LIFT} y el recall de "
            f"{CRITICAL_CLASS} es >= {USEFUL_EXTREME_RECALL}; limitado si el aumento de F1 es >= {LIMITED_F1_LIFT}; "
            "en otro caso, sin valor operativo. Es un criterio interno del proyecto, no un estándar clínico."
        ),
    }


@dataclass
class RandomForestRun:
    """Resultado de EA3: pipeline entrenado, tablas, datos de prueba y resumen serializable."""

    pipeline: Pipeline
    results: dict[str, Any]
    hyperparameter_comparison: pd.DataFrame
    metrics_by_class: pd.DataFrame
    permutation_importance: pd.DataFrame
    internal_importance: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray


def train_random_forest(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> RandomForestRun:
    """Entrena y evalúa el Random Forest sobre el DataFrame procesado.

    Args:
        df: DataFrame limpio (columnas en ``snake_case``, ``age_midpoint`` creada).
        test_size: proporción de prueba (la división es estratificada).
        random_state: semilla de toda la ejecución.
    """
    missing = [c for c in [TARGET, *FEATURES] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en el DataFrame procesado: {missing}")
    leaked = [c for c in FEATURES if c in EXCLUDED_COLUMNS]
    if leaked:
        raise ValueError(f"Variables excluidas por riesgo de fuga usadas como predictoras: {leaked}")

    data = df.dropna(subset=[TARGET])
    X, y = data[FEATURES], data[TARGET]
    labels = [c for c in CLASS_ORDER if c in set(y)] + sorted(set(y) - set(CLASS_ORDER))
    if CRITICAL_CLASS not in labels:
        raise ValueError(f"La clase crítica {CRITICAL_CLASS!r} no existe en el objetivo.")

    distribution = y.value_counts(normalize=True).reindex(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    comparison = compare_hyperparameters(
        X_train, y_train, search_rows=config.RF_SEARCH_ROWS, final_rows=len(X_train),
        random_state=random_state,
    )
    best_params, rationale = select_configuration(comparison)

    X_fit, y_fit = X_train, y_train  # se entrena con toda la partición de entrenamiento
    pipeline = build_pipeline(best_params, random_state)
    start = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start

    y_pred = pipeline.predict(X_test)
    metrics = _class_metrics(y_test, y_pred, labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    baseline_model = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
    baseline = _class_metrics(y_test, baseline_model.predict(X_test), labels)
    baseline["majority_class"] = str(baseline_model.classes_[np.argmax(baseline_model.class_prior_)])

    ext = labels.index(CRITICAL_CLASS)
    extreme_row = cm[ext]
    extreme_confusion = {
        "total": int(extreme_row.sum()),
        "correctos": int(extreme_row[ext]),
        "clasificados_como": {labels[j]: int(extreme_row[j]) for j in range(len(labels)) if j != ext},
    }

    perm_X, perm_y = X_test, y_test
    perm = permutation_importance(
        pipeline, perm_X, perm_y, scoring="f1_weighted", n_repeats=3,
        random_state=random_state, n_jobs=1,
    )
    perm_df = (
        pd.DataFrame(
            {"feature": FEATURES, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    internal = _group_internal_importance(pipeline)

    metrics_by_class = pd.DataFrame(metrics["per_class"]).T.rename_axis("class").reset_index()
    best_row = metrics_by_class.sort_values("recall", ascending=False).iloc[0]
    worst_row = metrics_by_class.sort_values("recall").iloc[0]

    results: dict[str, Any] = {
        "business_question": BUSINESS_QUESTION,
        "target": TARGET,
        "classes": labels,
        "class_distribution": {c: float(distribution[c]) for c in labels},
        "learning_type": "Aprendizaje supervisado de clasificación multiclase",
        "algorithm": "RandomForestClassifier dentro de un Pipeline (imputación + OneHotEncoder)",
        "features": FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "hyperparameters_selected": {k: v for k, v in best_params.items() if k != "name"} | {
            "config": best_params["name"], "random_state": random_state, "n_jobs": -1,
        },
        "selection_rationale": rationale,
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratified": True,
            "n_rows_modeling": int(len(data)),
            "n_train_available": int(len(X_train)),
            "n_train_used": int(len(X_fit)),
            "n_test": int(len(X_test)),
            "train_sampling": "entrenamiento con el 100 % de la partición de entrenamiento",
            "search_rows": int(min(config.RF_SEARCH_ROWS, len(X_train))),
        },
        "timing_seconds": {
            "hyperparameter_search": float(comparison["fit_seconds"].sum()),
            "final_training": float(train_seconds),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "per_class": metrics["per_class"],
        "recall_extreme": metrics["per_class"][CRITICAL_CLASS]["recall"],
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "extreme_confusion": extreme_confusion,
        "baseline": {
            "strategy": "most_frequent",
            "majority_class": baseline["majority_class"],
            "accuracy": baseline["accuracy"],
            "f1_weighted": baseline["f1_weighted"],
            "recall_extreme": baseline["per_class"][CRITICAL_CLASS]["recall"],
        },
        "best_recognized_class": {"class": str(best_row["class"]), "recall": float(best_row["recall"])},
        "worst_recognized_class": {"class": str(worst_row["class"]), "recall": float(worst_row["recall"])},
        "utility_assessment": _assess_utility(metrics, baseline),
        "permutation_importance": {
            "scoring": "f1_weighted",
            "n_repeats": 3,
            "rows_used": int(len(perm_X)),
            "values": perm_df.round(6).to_dict(orient="records"),
        },
        "internal_importance_grouped": {k: float(v) for k, v in internal.round(6).items()},
        "model_path": None,
    }
    return RandomForestRun(pipeline, results, comparison, metrics_by_class, perm_df, internal, y_test, y_pred)


def export_model(run: RandomForestRun, path: str | Path = config.RF_MODEL_PKL) -> Path:
    """Exporta el pipeline entrenado con joblib (comprimido) y registra su ruta y tamaño."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(run.pipeline, target, compress=3)
    size_mb = target.stat().st_size / 1e6
    if size_mb > config.RF_MAX_MODEL_MB + 10:
        raise RuntimeError(f"El modelo exportado pesa {size_mb:.1f} MB y excede el límite del proyecto.")
    run.results["model_path"] = (
        target.relative_to(config.PROJECT_ROOT).as_posix()
        if target.is_relative_to(config.PROJECT_ROOT)
        else str(target)
    )
    run.results["model_size_mb"] = round(size_mb, 2)
    return target


BUSINESS_QUESTION = (
    "Al momento de la admisión de un paciente, ¿con qué desempeño puede un modelo predecir su nivel de "
    "gravedad (Severity of Illness: Minor, Moderate, Extreme) a partir de información demográfica, "
    "antecedentes de salud, tipo de ingreso y variables logísticas disponibles, para apoyar el triaje "
    "preliminar y la planificación anticipada de camas y personal?"
)
