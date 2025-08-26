from __future__ import annotations
from pathlib import Path
import os, sys, platform
from platformdirs import user_cache_dir, user_data_dir

APP_NAME = "fast-stt"


def get_user_models_dir() -> Path:
    base = Path(user_data_dir(APP_NAME, appauthor=False))
    d = base / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model_file(model_name: str | None) -> Path:
    short = (model_name or "model").split("/")[-1]
    return get_user_models_dir() / f"{short}.nemo"


def get_gradio_tmp_dir() -> Path:
    # cache dir is safer for ephemeral files
    g = Path(user_cache_dir(APP_NAME, appauthor=False)) / "gradio"
    g.mkdir(parents=True, exist_ok=True)
    return g


__all__ = ["get_model_file", "get_gradio_tmp_dir", "get_user_models_dir", "APP_NAME"]
