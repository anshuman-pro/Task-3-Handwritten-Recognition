"""
Training pipeline.

Trains the CNN with early stopping, learning-rate scheduling and model
checkpointing, then renders the accuracy/loss curves. The best weights (by
validation accuracy) are saved to ``models/best_model.keras`` and loaded
automatically by ``test.py`` / ``predict.py`` / the app.

Run:
    python -m src.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import DataLoader
from src.model import CNNModel
from src.utils import (
    Config,
    PROJECT_ROOT,
    configure_gpu,
    ensure_dir,
    get_logger,
    load_config,
    set_seed,
)

LOGGER = get_logger("train", log_dir="logs")
BEST_MODEL_NAME = "best_model.keras"


class Trainer:
    """Orchestrates CNN training, checkpointing and curve plotting."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.models_dir = ensure_dir(config.get("training.models_dir", "models"))
        self.outputs_dir = ensure_dir(config.get("training.outputs_dir", "outputs"))

    def _callbacks(self):
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
        )

        best_path = Path(self.models_dir) / BEST_MODEL_NAME
        return [
            EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.get("training.early_stopping_patience", 5),
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.get("training.lr_factor", 0.5),
                patience=self.config.get("training.lr_patience", 2),
                min_lr=self.config.get("training.min_lr", 1e-5),
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(best_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

    def train(self):
        set_seed(self.config.get("project.random_seed", 42))
        configure_gpu(LOGGER)

        data = DataLoader(self.config).load()
        model = CNNModel(self.config).build_and_compile()
        model.summary(print_fn=LOGGER.info)

        history = model.fit(
            data.x_train, data.y_train,
            validation_data=(data.x_val, data.y_val),
            epochs=self.config.get("training.epochs", 20),
            batch_size=self.config.get("training.batch_size", 128),
            callbacks=self._callbacks(),
            verbose=2,
        )

        self._plot_history(history.history)
        self._save_history(history.history)

        val_loss, val_acc = model.evaluate(data.x_val, data.y_val, verbose=0)
        LOGGER.info("Best model | val_accuracy=%.4f val_loss=%.4f", val_acc, val_loss)
        LOGGER.info("Saved best model -> %s", Path(self.models_dir) / BEST_MODEL_NAME)
        return history

    # ------------------------------------------------------------------ #
    def _plot_history(self, history: dict) -> None:
        epochs = range(1, len(history["accuracy"]) + 1)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(epochs, history["accuracy"], "o-", label="Train")
        ax.plot(epochs, history["val_accuracy"], "s-", label="Validation")
        ax.set_title("Model Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "training_accuracy.png")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(epochs, history["loss"], "o-", label="Train")
        ax.plot(epochs, history["val_loss"], "s-", label="Validation")
        ax.set_title("Model Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, "training_loss.png")

    def _save_history(self, history: dict) -> None:
        path = Path(self.outputs_dir) / "history.json"
        serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = Path(self.outputs_dir) / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved figure -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the handwriting CNN.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    Trainer(load_config(args.config)).train()


if __name__ == "__main__":
    main()
