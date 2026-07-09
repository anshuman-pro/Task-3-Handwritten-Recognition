"""
Dataset loading and preparation.

Loads MNIST via Keras, carves a validation split out of the training set and
returns model-ready ``float32`` tensors with an explicit channel dimension.
Normalization and augmentation live inside the model graph (see ``model.py``),
so the arrays returned here are raw pixel intensities in ``[0, 255]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import Config, get_logger

LOGGER = get_logger("dataset")


@dataclass
class Dataset:
    """Container for the train / validation / test partitions."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    class_names: list[str]


class DataLoader:
    """Loads and partitions the handwritten-character dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.dataset_name = config.get("data.dataset", "mnist")
        self.val_size = float(config.get("data.val_size", 0.1))
        self.seed = config.get("project.random_seed", 42)

    def load(self) -> Dataset:
        if self.dataset_name != "mnist":
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'. Use 'mnist'.")

        from tensorflow.keras.datasets import mnist

        LOGGER.info("Loading MNIST via Keras...")
        (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=self.val_size, stratify=y_train_full, random_state=self.seed,
        )

        x_train = self._reshape(x_train)
        x_val = self._reshape(x_val)
        x_test = self._reshape(x_test)

        class_names = [str(i) for i in range(10)]
        LOGGER.info(
            "Prepared MNIST -> train=%d, val=%d, test=%d | shape=%s",
            len(x_train), len(x_val), len(x_test), x_train.shape[1:],
        )
        return Dataset(
            x_train=x_train, y_train=y_train.astype("int64"),
            x_val=x_val, y_val=y_val.astype("int64"),
            x_test=x_test, y_test=y_test.astype("int64"),
            num_classes=int(self.config.get("data.num_classes", 10)),
            class_names=class_names,
        )

    @staticmethod
    def _reshape(images: np.ndarray) -> np.ndarray:
        """Add a trailing channel dimension and cast to float32."""
        return images.reshape(-1, 28, 28, 1).astype("float32")
