"""Generación de reportes de texto a partir de DataFrames y resultados de modelos."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

SEP = "=" * 70


class DatasetSummarizer:
    """Resume un DataFrame en un archivo ``.txt`` (``reports/data-summary.txt``)."""

    def __init__(self, title: str = "RESUMEN DEL DATAFRAME") -> None:
        self.title = title

    def generate(self, df: pd.DataFrame) -> str:
        """Construye el texto del resumen."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        if df.empty:
            raise ValueError("El DataFrame está vacío.")
        numeric = df.select_dtypes(include="number")
        lines = [
            self.title,
            SEP,
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Filas duplicadas: {int(df.duplicated().sum())}",
            "",
            "Nombres de columnas:",
            ", ".join(df.columns),
            "",
            "Tipos de datos:",
            *[f"- {c}: {t}" for c, t in df.dtypes.items()],
            "",
            "Valores nulos por columna:",
            *[f"- {c}: {n}" for c, n in df.isna().sum().items()],
        ]
        if not numeric.empty:
            lines += [
                "",
                "Estadísticas descriptivas (columnas numéricas):",
                numeric.describe().T.round(3).to_string(),
            ]
        return "\n".join(lines) + "\n"

    def save(self, df: pd.DataFrame, path: str | Path = config.DATA_SUMMARY_TXT) -> Path:
        """Escribe el resumen en ``path`` (crea la carpeta) y devuelve la ruta."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.generate(df), encoding="utf-8")
        return target


def _bullets(items: list[str], indent: str = "  - ") -> list[str]:
    return [f"{indent}{item}" for item in items]


class ModelSummaryReporter:
    """Compone ``reports/model-summary.txt`` solo con resultados reales de la ejecución."""

    def generate(
        self,
        metadata: dict[str, Any],
        regression: dict[str, dict[str, Any]],
        classification: dict[str, Any],
    ) -> str:
        """Construye el texto del resumen de ambos modelos."""
        lines = [
            "RESUMEN AUTOMÁTICO DE MODELOS - EA2",
            SEP,
            f"Fecha y hora de ejecución: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S UTC%z')}",
            "",
            "1. DATASET",
            "-" * 70,
            f"Identificador: {metadata['dataset_id']}",
            f"URL: {metadata['dataset_url']}",
            f"Configuración / split: {metadata['config']} / {metadata['split']}",
            f"Filas en Hugging Face: {metadata['filas_originales_en_hf']}",
            f"Muestreo: {metadata['muestreo']}",
            f"Dimensiones iniciales (filas x columnas): {metadata['filas_cargadas']} x {len(metadata['columnas_iniciales'])}",
            f"Columnas iniciales: {', '.join(metadata['columnas_iniciales'])}",
            f"Dimensiones tras el preprocesamiento: {metadata['filas_procesadas']} x {len(metadata['columnas_procesadas'])}",
            f"Columnas procesadas: {', '.join(metadata['columnas_procesadas'])}",
            "",
            "2. PREPROCESAMIENTO",
            "-" * 70,
            *_bullets(metadata["preprocesamiento"], "  * "),
        ]
        for key in ("simple", "multiple"):
            lines += self._regression_block(regression[key], f"{3 if key == 'simple' else 4}. ")
        lines += self._classification_block(classification, "5. ")
        lines += [
            "",
            "6. CONCLUSIONES Y LIMITACIONES",
            "-" * 70,
            *_bullets(
                [
                    f"Regresión múltiple: R²={regression['multiple']['metrics']['R2']:.4f} "
                    f"frente a R²={regression['simple']['metrics']['R2']:.4f} de la simple.",
                    f"Clasificación: exactitud={classification['metrics']['accuracy']:.4f}, "
                    f"F1 ponderado={classification['metrics']['f1_weighted']:.4f}.",
                    *regression["multiple"]["limitations"],
                    *classification["limitations"],
                ],
                "  * ",
            ),
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _model_header(res: dict[str, Any], number: str) -> list[str]:
        sp = res["split"]
        return [
            "",
            f"{number}{res['name'].upper()}",
            "-" * 70,
            f"Pregunta: {res['question']}",
            f"Variable objetivo: {res['target']}",
            f"Variables predictoras: {', '.join(res['predictors'])}",
            f"Algoritmo: {res['algorithm']}",
            f"Parámetros: {json.dumps(res['parameters'], ensure_ascii=False)}",
            f"División train/test: test_size={sp['test_size']}, random_state={sp['random_state']}, "
            f"n_train={sp['n_train']}, n_test={sp['n_test']}"
            + (", estratificada" if sp.get("stratified") else ""),
        ]

    def _regression_block(self, res: dict[str, Any], number: str) -> list[str]:
        m = res["metrics"]
        return [
            *self._model_header(res, number),
            "Métricas en prueba:",
            f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
            f"(MAE de la media={m['MAE_baseline_media']:.4f})",
            f"Intercepto: {res['intercept']:.4f}",
            "Coeficientes:",
            *[f"  {k}: {v:.4f}" for k, v in res["coefficients"].items()],
            "Interpretación:",
            *_bullets(res["interpretation"]),
        ]

    def _classification_block(self, res: dict[str, Any], number: str) -> list[str]:
        m = res["metrics"]
        labels = res["confusion_matrix"]["labels"]
        matrix = res["confusion_matrix"]["matrix"]
        width = max(len(label) for label in labels + ["real \\ pred"]) + 2
        header = "real \\ pred".ljust(width) + "".join(label.rjust(width) for label in labels)
        rows = [
            labels[i].ljust(width) + "".join(str(v).rjust(width) for v in row)
            for i, row in enumerate(matrix)
        ]
        return [
            *self._model_header(res, number),
            f"Clases: {', '.join(labels)}",
            f"Justificación de métricas: {res['metrics_justification']}",
            "Métricas en prueba:",
            f"  accuracy={m['accuracy']:.4f}  precision_ponderada={m['precision_weighted']:.4f}  "
            f"recall_ponderado={m['recall_weighted']:.4f}  f1_ponderado={m['f1_weighted']:.4f}  "
            f"(exactitud clase mayoritaria={m['accuracy_baseline_mayoritaria']:.4f})",
            "Matriz de confusión (filas=real, columnas=predicho):",
            "  " + header,
            *["  " + r for r in rows],
            "Reporte por clase:",
            *[
                f"  {c}: precision={v['precision']:.3f} recall={v['recall']:.3f} "
                f"f1={v['f1-score']:.3f} soporte={int(v['support'])}"
                for c, v in res["per_class"].items()
            ],
            "Variables más importantes:",
            *[f"  {d['feature']}: {d['importance']:.4f}" for d in res["top_feature_importances"][:5]],
            "Interpretación:",
            *_bullets(res["interpretation"]),
        ]

    def save(
        self,
        metadata: dict[str, Any],
        regression: dict[str, dict[str, Any]],
        classification: dict[str, Any],
        path: str | Path = config.MODEL_SUMMARY_TXT,
    ) -> Path:
        """Escribe ``model-summary.txt`` y devuelve su ruta."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.generate(metadata, regression, classification), encoding="utf-8")
        return target


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Guarda un diccionario serializable como JSON UTF-8."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Lee un JSON; error claro si el paso previo no se ejecutó."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"No existe {target}. Ejecute antes el notebook/paso que lo genera."
        )
    return json.loads(target.read_text(encoding="utf-8"))
