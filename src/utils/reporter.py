"""Generación de reportes de texto a partir de DataFrames y resultados de modelos."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.models.train_random_forest import CRITICAL_CLASS, RandomForestRun  # noqa: E402

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


RF_SEP = "=" * 72
INSTITUTION = "Hospital San Juan de Dios de Rionegro (caso hipotético)"
DISCLAIMER = (
    "Este modelo es un apoyo administrativo y preliminar. NO es un diagnóstico ni una decisión clínica "
    "autónoma: requiere validación clínica, revisión de sesgos y supervisión humana."
)


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _fmt_cm(labels: list[str], matrix: list[list[int]]) -> list[str]:
    width = max(len(x) for x in [*labels, "real \\ pred"]) + 2
    lines = ["real \\ pred".ljust(width) + "".join(l.rjust(width) for l in labels)]
    lines += [labels[i].ljust(width) + "".join(str(v).rjust(width) for v in row) for i, row in enumerate(matrix)]
    return lines


class RandomForestReporter:
    """Genera todos los entregables del Random Forest a partir de una ejecución."""

    def __init__(
        self, run: RandomForestRun, metadata: dict[str, Any], preprocessing_evidence: pd.DataFrame
    ) -> None:
        self.run = run
        self.res = run.results
        self.meta = metadata
        self.evidence = preprocessing_evidence

    # ------------------------------------------------------------- figuras
    def save_confusion_matrix(self, path: Path = config.RF_CONFUSION_PNG) -> Path:
        labels = self.res["confusion_matrix"]["labels"]
        cm = np.array(self.res["confusion_matrix"]["matrix"])
        pct = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.0%}", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black", fontsize=10)
        ax.set(xlabel="Severidad predicha", ylabel="Severidad real",
               title="Matriz de confusión (prueba)\nconteo y % de cada clase real")
        fig.colorbar(im, ax=ax, fraction=0.046, label="% de la clase real")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def save_feature_importance(self, path: Path = config.RF_IMPORTANCE_PNG) -> Path:
        perm = self.run.permutation_importance.sort_values("importance_mean")
        internal = self.run.internal_importance.reindex(perm["feature"]).fillna(0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        axes[0].barh(perm["feature"], perm["importance_mean"], xerr=perm["importance_std"], color="#2a6f97")
        axes[0].axvline(0, color="grey", lw=0.8)
        axes[0].set(title="Importancia por permutación (prueba)", xlabel="Caída del F1 ponderado al permutar")
        axes[1].barh(internal.index, internal.values, color="#8ab17d")
        axes[1].set(title="Importancia interna (agrupada)", xlabel="Reducción media de impureza")
        fig.suptitle("Importancia de variables - Random Forest", fontsize=12)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # -------------------------------------------------------------- tablas
    def save_tables(self) -> dict[str, Path]:
        config.RF_DIR.mkdir(parents=True, exist_ok=True)
        self.run.hyperparameter_comparison.to_csv(config.RF_HYPERPARAMETERS_CSV, index=False)
        self.run.metrics_by_class.round(4).to_csv(config.RF_METRICS_BY_CLASS_CSV, index=False)
        return {
            "hyperparameters": config.RF_HYPERPARAMETERS_CSV,
            "metrics_by_class": config.RF_METRICS_BY_CLASS_CSV,
        }

    def save_results_json(self) -> Path:
        payload = {
            **self.res,
            "dataset": {k: v for k, v in self.meta.items() if k != "csv_guardados"},
            "preprocessing_evidence": self.evidence.astype(str).to_dict(orient="records"),
            "hyperparameter_comparison": self.run.hyperparameter_comparison.to_dict(orient="records"),
            "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        return save_json(payload, config.RF_RESULTS_JSON)

    # ---------------------------------------------------- resumen de texto
    def summary_text(self) -> str:
        r, m = self.res, self.meta
        sp, met, base = r["split"], r["metrics"], r["baseline"]
        hp = r["hyperparameters_selected"]
        util = r["utility_assessment"]
        ec = r["extreme_confusion"]
        lines = [
            "RESUMEN AUTOMÁTICO - RANDOM FOREST",
            RF_SEP,
            f"Fecha y hora de ejecución: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S UTC%z')}",
            f"Institución / caso: {INSTITUTION}",
            "",
            "1. PREGUNTA DE NEGOCIO",
            "-" * 72,
            r["business_question"],
            "",
            "2. DATASET",
            "-" * 72,
            f"Identificador: {m['dataset_id']}",
            f"URL: {m['dataset_url']}",
            f"Configuración / split: {m['config']} / {m['split']}",
            f"Tamaño del DataFrame original: {m['filas_cargadas']} x {len(m['columnas_iniciales'])}",
            f"Tamaño tras el preprocesamiento: {m['filas_procesadas']} x {len(m['columnas_procesadas'])}",
            f"Columnas utilizadas como predictoras: {', '.join(r['features'])}",
            f"Variable objetivo: {r['target']} (clases: {', '.join(r['classes'])})",
            "Distribución de clases: " + ", ".join(f"{c}={v:.1%}" for c, v in r["class_distribution"].items()),
            "",
            "3. PREPROCESAMIENTO",
            "-" * 72,
            *[f"  * {s}" for s in m["preprocesamiento"]],
            "Evidencia antes / después:",
            *self._evidence_lines(),
            "",
            "4. VARIABLES EXCLUIDAS (prevención de fuga de información)",
            "-" * 72,
            *[f"  * {c}: {why}" for c, why in r["excluded_columns"].items()],
            "",
            "5. MODELO",
            "-" * 72,
            f"Tipo de aprendizaje: {r['learning_type']}",
            f"Algoritmo: {r['algorithm']}",
            "Random Forest entrena muchos árboles de decisión sobre muestras aleatorias de los datos y de "
            "las variables, y la clase final se decide por votación. No se escalan las variables porque "
            "los árboles dividen por umbrales y no dependen de la escala.",
            f"Configuraciones evaluadas (submuestra de {sp['search_rows']} filas, validación 25 %):",
            self.run.hyperparameter_comparison.to_string(index=False),
            f"Criterio de selección: {r['selection_rationale']}",
            f"Hiperparámetros seleccionados: {hp}",
            f"Tiempos: búsqueda={r['timing_seconds']['hyperparameter_search']:.1f} s, "
            f"entrenamiento final={r['timing_seconds']['final_training']:.1f} s",
            f"División train/test: {int((1 - sp['test_size']) * 100)}/{int(sp['test_size'] * 100)} estratificada, "
            f"random_state={sp['random_state']}",
            f"Filas: modelado={sp['n_rows_modeling']}, train={sp['n_train_available']}, "
            f"test={sp['n_test']}",
            f"Muestreo de entrenamiento: {sp['train_sampling']}.",
            f"Modelo exportado: {r.get('model_path')} ({r.get('model_size_mb')} MB)",
            "",
            "6. MÉTRICAS EN PRUEBA",
            "-" * 72,
            f"accuracy={met['accuracy']:.4f}  precision_ponderada={met['precision_weighted']:.4f}  "
            f"recall_ponderado={met['recall_weighted']:.4f}  f1_ponderado={met['f1_weighted']:.4f}",
            "Por clase:",
            *[
                f"  {c}: precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} soporte={v['support']}"
                for c, v in r["per_class"].items()
            ],
            f"Recall de {CRITICAL_CLASS}: {r['recall_extreme']:.4f}",
            "Matriz de confusión (filas=real, columnas=predicho):",
            *["  " + x for x in _fmt_cm(r["confusion_matrix"]["labels"], r["confusion_matrix"]["matrix"])],
            "",
            "7. LÍNEA BASE (siempre la clase mayoritaria)",
            "-" * 72,
            f"Clase mayoritaria={base['majority_class']}  accuracy={base['accuracy']:.4f}  "
            f"f1_ponderado={base['f1_weighted']:.4f}  recall_{CRITICAL_CLASS}={base['recall_extreme']:.4f}",
            f"Diferencia del modelo vs. línea base: accuracy={met['accuracy'] - base['accuracy']:+.4f}, "
            f"f1_ponderado={util['f1_lift_vs_baseline']:+.4f}",
            "",
            "8. VARIABLES MÁS RELEVANTES",
            "-" * 72,
            f"Importancia por permutación (F1 ponderado; {r['permutation_importance']['rows_used']} filas de prueba, "
            f"{r['permutation_importance']['n_repeats']} repeticiones):",
            *[
                f"  {d['feature']}: {d['importance_mean']:.5f} (+/- {d['importance_std']:.5f})"
                for d in r["permutation_importance"]["values"]
            ],
            "",
            "9. INTERPRETACIÓN",
            "-" * 72,
            *[f"  * {x}" for x in self.interpretation()],
            "",
            "10. LIMITACIONES",
            "-" * 72,
            *[f"  * {x}" for x in self.limitations()],
            "",
            DISCLAIMER,
        ]
        return "\n".join(lines) + "\n"

    def _evidence_lines(self) -> list[str]:
        lines = []
        for row in self.evidence.itertuples(index=False):
            before, after = str(row.antes), str(row.despues)
            if before and after:
                lines.append(f"  - {row.metrica}: {before} -> {after}")
            else:
                lines.append(f"  - {row.metrica}: {before or after}")
        return lines

    def interpretation(self) -> list[str]:
        r = self.res
        met, base, util, ec = r["metrics"], r["baseline"], r["utility_assessment"], r["extreme_confusion"]
        best, worst = r["best_recognized_class"], r["worst_recognized_class"]
        miss = ", ".join(f"{n} como {c}" for c, n in ec["clasificados_como"].items())
        top = r["permutation_importance"]["values"][0]
        positive = [
            d for d in r["permutation_importance"]["values"] if d["importance_mean"] > 2 * d["importance_std"]
        ]
        return [
            f"Clase mejor identificada: {best['class']} (recall={best['recall']:.3f}); peor identificada: "
            f"{worst['class']} (recall={worst['recall']:.3f}).",
            f"De {ec['total']} pacientes {CRITICAL_CLASS} reales en prueba, el modelo acertó {ec['correctos']} "
            f"({r['recall_extreme']:.1%}) y confundió {miss}.",
            f"Frente a la línea base: accuracy {met['accuracy']:.4f} vs {base['accuracy']:.4f} y F1 ponderado "
            f"{met['f1_weighted']:.4f} vs {base['f1_weighted']:.4f}.",
            f"Valoración de utilidad operativa: {util['level']}. Criterio: {util['criteria']}",
            (
                f"La variable con mayor importancia por permutación es {top['feature']} "
                f"({top['importance_mean']:.5f}); {len(positive)} de {len(r['features'])} variables tienen "
                "un aporte claramente distinguible del ruido (media > 2 desviaciones)."
                if positive
                else "Ninguna variable muestra un aporte medible por permutación: las predictoras disponibles "
                "casi no contienen señal sobre la severidad."
            ),
        ]

    def limitations(self) -> list[str]:
        r = self.res
        return [
            "Las variables disponibles al ingreso tienen poder discriminante limitado; el desempeño medido "
            "no debe extrapolarse a otros hospitales o periodos sin validación local.",
            "El dataset es público y de origen no verificado; podría no representar a la población del hospital.",
            "Se asume que visitors_with_patient, staff_available y available_extra_rooms_in_hospital se conocen "
            "en el momento de la admisión; si no, habría que retirarlas.",
            "La búsqueda de hiperparámetros se hizo en una submuestra estratificada de "
            f"{r['split']['search_rows']:,} filas por tiempo de cómputo; el modelo final usa todo el entrenamiento.",
            "Los modelos con hojas pequeñas (más pesados) se excluyeron de la selección para que el modelo "
            "exportado pese menos de lo permitido por GitHub.",
            "No se evaluó equidad por género, edad o seguro; antes de cualquier uso real se requiere auditoría de sesgos.",
            "Las importancias por permutación son asociaciones predictivas, no causas clínicas.",
            "No debe usarse como diagnóstico ni para decidir tratamiento: es solo apoyo administrativo preliminar.",
        ]

    def save_summary(self, path: Path = config.RF_SUMMARY_TXT) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.summary_text(), encoding="utf-8")
        return path

    # ------------------------------------------------------- hallazgos (md)
    def findings_markdown(self, out_path: Path = config.RF_FINDINGS_MD) -> str:
        r, base = self.res, self.res["baseline"]
        met, util, ec = r["metrics"], r["utility_assessment"], r["extreme_confusion"]
        cm_png = _rel(config.RF_CONFUSION_PNG, out_path.parent)
        fi_png = _rel(config.RF_IMPORTANCE_PNG, out_path.parent)
        best, worst = r["best_recognized_class"], r["worst_recognized_class"]
        sp = r["split"]
        perm = r["permutation_importance"]["values"]
        top3 = ", ".join(f"{d['feature']} ({d['importance_mean']:.4f})" for d in perm[:3])
        miss = "; ".join(f"{n:,} como {c}" for c, n in ec["clasificados_como"].items())
        verdict = {
            "apto para apoyo operativo preliminar": "El modelo **puede apoyar de forma preliminar** la planificación, siempre con revisión humana.",
            "utilidad limitada": "El modelo **mejora de forma modesta** a una regla trivial; sirve, como mucho, como una señal complementaria débil.",
        }.get(util["level"], "El modelo **no aporta una mejora operativa suficiente** frente a una regla trivial; con la información actual no es recomendable usarlo para planificar.")
        lines = [
            "# Hallazgos: predicción de severidad con Random Forest",
            "",
            f"_{INSTITUTION} · Generado automáticamente el {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}_",
            "",
            "## 1. Pregunta de negocio",
            "",
            f"> {r['business_question']}",
            "",
            "## 2. Respuesta",
            "",
            f"{verdict} Con los datos de admisión disponibles, el modelo acierta la severidad en "
            f"**{met['accuracy']:.1%}** de los pacientes de prueba; una regla que siempre dijera "
            f"«{base['majority_class']}» acertaría **{base['accuracy']:.1%}**. Al ponderar por clase (F1), el modelo "
            f"obtiene **{met['f1_weighted']:.3f}** frente a **{base['f1_weighted']:.3f}** de la regla trivial "
            f"(diferencia {util['f1_lift_vs_baseline']:+.3f}). Valoración según el criterio interno del proyecto: "
            f"**{util['level']}**.",
            "",
            "## 3. Hallazgos principales",
            "",
            f"- Se evaluó en {sp['n_test']:,} pacientes de prueba que el modelo no vio al entrenar.",
            f"- Clase mejor identificada: **{best['class']}** (recall {best['recall']:.1%}); la más difícil: "
            f"**{worst['class']}** (recall {worst['recall']:.1%}).",
            f"- De {ec['total']:,} pacientes realmente **{CRITICAL_CLASS}**, el modelo detectó {ec['correctos']:,} "
            f"({r['recall_extreme']:.1%}) y no detectó {ec['total'] - ec['correctos']:,} ({miss}).",
            f"- Variables con mayor aporte (importancia por permutación): {top3}.",
            "",
            "## 4. Cómo leer las gráficas",
            "",
            f"![Matriz de confusión]({cm_png})",
            "",
            "Cada fila es la severidad real y cada columna la que predijo el modelo. La diagonal son los aciertos; "
            "fuera de ella están los errores. El porcentaje indica qué parte de cada clase real cayó en cada predicción.",
            "",
            f"![Importancia de variables]({fi_png})",
            "",
            "Izquierda: cuánto empeora el modelo si se «desordena» una variable (cuanto más baja el F1, más la usa el modelo). "
            "Derecha: importancia interna del bosque, agrupando las categorías de una misma variable. "
            "Valores cercanos a cero indican que la variable casi no ayuda a predecir.",
            "",
            "## 5. Confiabilidad del resultado",
            "",
            f"- División 80/20 estratificada con semilla fija ({sp['random_state']}); resultados reproducibles.",
            f"- Entrenamiento con {sp['n_train_available']:,} filas ({sp['train_sampling']}).",
            f"- Configuración elegida comparando {len(self.run.hyperparameter_comparison)} alternativas "
            f"({r['hyperparameters_selected']['config']}); criterio: F1 ponderado y recall de {CRITICAL_CLASS}.",
            "- Una sola partición de prueba: no se midió la variabilidad entre particiones.",
            "",
            "## 6. Errores y limitaciones",
            "",
            *[f"- {x}" for x in self.limitations()],
            "",
            "## 7. Recomendaciones operativas",
            "",
        ]
        if util["level"].startswith("apto"):
            lines += [
                "- Usar la predicción solo como alerta temprana para dimensionar camas y turnos, nunca para priorizar a un paciente individual sin valoración médica.",
                f"- Reforzar la revisión humana de los casos que el modelo no marca como {CRITICAL_CLASS}, porque se omiten {ec['total'] - ec['correctos']:,} de {ec['total']:,}.",
            ]
        else:
            lines += [
                "- **No** usar este modelo para planificar camas ni personal con la información actual.",
                f"- Mantener el triaje manual: el modelo omitiría {ec['total'] - ec['correctos']:,} de {ec['total']:,} pacientes {CRITICAL_CLASS}.",
                "- Evaluar variables adicionales disponibles al ingreso (signos vitales, motivo de consulta, escala de triaje) antes de repetir el estudio.",
            ]
        lines += [
            "- Validar con datos propios del hospital y auditar sesgos (género, edad, seguro) antes de cualquier piloto.",
            "- Repetir la evaluación periódicamente para detectar deterioro del desempeño.",
            "",
            "## 8. Advertencia",
            "",
            f"**{DISCLAIMER}**",
            "",
        ]
        return "\n".join(lines)

    def save_findings(self, path: Path = config.RF_FINDINGS_MD) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.findings_markdown(path), encoding="utf-8")
        return path

    def save_all(self) -> dict[str, Path]:
        """Genera figuras, tablas, JSON, resumen y hallazgos; devuelve las rutas."""
        paths = {
            "confusion_matrix": self.save_confusion_matrix(),
            "feature_importance": self.save_feature_importance(),
            **self.save_tables(),
            "results_json": self.save_results_json(),
            "summary_txt": self.save_summary(),
            "findings_md": self.save_findings(),
        }
        return paths
