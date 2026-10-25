"""Carga de datasets de Hugging Face como ``pandas.DataFrame``.

Esta clase solo se encarga de la conexión y la carga: no transforma los datos
ni genera reportes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)


def read_hf_token(env_file: str | Path | None = None) -> str | None:
    """Obtiene el token de Hugging Face desde ``HF_TOKEN`` o un archivo ``.env``.

    Si existe ``python-dotenv`` y un archivo ``.env`` (ver ``.env.example``), se
    cargan sus variables sin sobrescribir las ya definidas en el entorno.
    El token es opcional para datasets públicos, pero evita límites de uso.
    """
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(env_file or find_dotenv(usecwd=True), override=False)
    except ImportError:  # python-dotenv es opcional
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token or None


class HFDatasetLoader:
    """Obtiene un dataset de Hugging Face y lo devuelve como DataFrame.

    Args:
        dataset_id: Identificador ``usuario/dataset`` en Hugging Face.
        config: Configuración del dataset (``None`` usa la predeterminada).
        split: Split que se carga (por ejemplo ``"train"``).
        token: Token de Hugging Face. Si es ``None`` se lee con
            :func:`read_hf_token`.
        cache_dir: Directorio de caché de ``datasets`` (opcional).
        max_rows: Si se indica, devuelve una muestra aleatoria reproducible de
            este tamaño máximo.
        seed: Semilla del muestreo aleatorio.
    """

    def __init__(
        self,
        dataset_id: str,
        config: str | None = None,
        split: str = "train",
        token: str | None = None,
        cache_dir: str | Path | None = None,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        if not dataset_id or "/" not in dataset_id:
            raise ValueError(
                f"dataset_id inválido: {dataset_id!r}. Use el formato 'usuario/dataset'."
            )
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows debe ser un entero positivo.")
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.token = token if token is not None else read_hf_token()
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.max_rows = max_rows
        self.seed = seed
        self.df: pd.DataFrame | None = None
        self.total_rows: int | None = None

    def available_configs(self) -> list[str]:
        """Lista las configuraciones reales del dataset."""
        try:
            return get_dataset_config_names(self.dataset_id, token=self.token)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo consultar '{self.dataset_id}' en Hugging Face: {exc}"
            ) from exc

    def available_splits(self) -> dict[str, list[str]]:
        """Devuelve ``{configuración: [splits]}`` consultando Hugging Face."""
        configs = self.available_configs()
        try:
            return {
                cfg: get_dataset_split_names(self.dataset_id, cfg, token=self.token)
                for cfg in configs
            }
        except Exception as exc:
            raise RuntimeError(f"No se pudieron obtener los splits: {exc}") from exc

    def load(self) -> pd.DataFrame:
        """Descarga el split indicado y lo devuelve como DataFrame."""
        splits = self.available_splits()
        config = self.config or ("default" if "default" in splits else next(iter(splits)))
        if config not in splits:
            raise ValueError(
                f"Configuración {config!r} inexistente. Disponibles: {sorted(splits)}."
            )
        if self.split not in splits[config]:
            raise ValueError(
                f"Split {self.split!r} inexistente en {config!r}. "
                f"Disponibles: {splits[config]}."
            )
        try:
            dataset: Dataset = load_dataset(
                self.dataset_id,
                self.config,
                split=self.split,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falló la descarga de '{self.dataset_id}' (split={self.split}): {exc}"
            ) from exc

        self.total_rows = len(dataset)
        if self.max_rows is not None and self.max_rows < len(dataset):
            dataset = dataset.shuffle(seed=self.seed).select(range(self.max_rows))
        self.df = dataset.to_pandas()
        if self.df.empty:
            raise RuntimeError("El dataset se cargó pero no contiene filas.")
        return self.df

    def info(self) -> dict[str, object]:
        """Metadatos de la última carga (requiere llamar antes a ``load``)."""
        if self.df is None:
            raise RuntimeError("Llame a load() antes de info().")
        sampled = (
            self.max_rows is not None
            and self.total_rows is not None
            and self.max_rows < self.total_rows
        )
        return {
            "dataset_id": self.dataset_id,
            "config": self.config or "default",
            "split": self.split,
            "filas_originales_en_hf": self.total_rows,
            "filas_cargadas": len(self.df),
            "columnas": list(self.df.columns),
            "muestreo": (
                f"muestra aleatoria de {self.max_rows} filas (seed={self.seed})"
                if sampled
                else "dataset completo"
            ),
        }
