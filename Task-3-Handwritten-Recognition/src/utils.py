"""
Reusable utilities: configuration, logging, reproducibility, GPU setup and
custom exceptions. Dependency-light so every module can import it safely.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class RecognitionError(Exception):
    """Base exception for the handwritten-recognition project."""


class ModelError(RecognitionError):
    """Raised for model build / load / inference failures."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    raw: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise RecognitionError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))


# --------------------------------------------------------------------------- #
# Reproducibility & hardware
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def configure_gpu(logger: logging.Logger | None = None) -> str:
    """Enable memory growth on any available GPU; fall back to CPU otherwise."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = f"{len(gpus)} GPU(s)" if gpus else "CPU"
    except Exception:  # pragma: no cover - hardware/driver dependent
        device = "CPU"
    if logger:
        logger.info("Compute device: %s", device)
    return device


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: str = "INFO", log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_dir:
        log_path = PROJECT_ROOT / log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike) -> Path:
    full = PROJECT_ROOT / path if not os.path.isabs(str(path)) else Path(path)
    full.mkdir(parents=True, exist_ok=True)
    return full
