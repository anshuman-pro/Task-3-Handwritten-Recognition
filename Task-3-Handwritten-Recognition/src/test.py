"""
Evaluation on the held-out test set.

Loads the best saved model and produces the test accuracy, a classification
report, a confusion matrix and a grid of sample predictions (correct in green,
incorrect in red).

Run:
    python -m src.test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import DataLoader
from src.utils import (
    Config,
    ModelError,
    PROJECT_ROOT,
    ensure_dir,
    get_logger,
    load_config,
    set_seed,
)

LOGGER = get_logger("test")
sns.set_theme(style="white")
BEST_MODEL_NAME = "best_model.keras"


class Evaluator:
    """Evaluates the trained CNN and generates reports and figures."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.outputs_dir = Path(ensure_dir(config.get("training.outputs_dir", "outputs")))
        self.models_dir = Path(config.get("training.models_dir", "models"))

    def _load_model(self):
        from tensorflow.keras.models import load_model

        path = PROJECT_ROOT / self.models_dir / BEST_MODEL_NAME
        if not path.exists():
            raise ModelError(f"Trained model not found at {path}. Run `python -m src.train` first.")
        LOGGER.info("Loading model from %s", path)
        return load_model(path)

    def evaluate(self) -> dict:
        set_seed(self.config.get("project.random_seed", 42))
        data = DataLoader(self.config).load()
        model = self._load_model()

        loss, acc = model.evaluate(data.x_test, data.y_test, verbose=0)
        LOGGER.info("Test accuracy = %.4f | Test loss = %.4f", acc, loss)

        proba = model.predict(data.x_test, verbose=0)
        y_pred = proba.argmax(axis=1)

        report = classification_report(
            data.y_test, y_pred, target_names=data.class_names, digits=4
        )
        (self.outputs_dir / "classification_report.txt").write_text(
            f"Test accuracy: {acc:.4f}\n\n{report}\n", encoding="utf-8"
        )
        LOGGER.info("Classification report:\n%s", report)

        self._plot_confusion_matrix(data.y_test, y_pred, data.class_names)
        self._plot_predictions(data.x_test, data.y_test, y_pred, proba, data.class_names)
        return {"test_accuracy": float(acc), "test_loss": float(loss)}

    def _plot_confusion_matrix(self, y_true, y_pred, class_names) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Test Set")
        self._save(fig, "confusion_matrix.png")

    def _plot_predictions(self, x_test, y_true, y_pred, proba, class_names, n: int = 20) -> None:
        rng = np.random.default_rng(self.config.get("project.random_seed", 42))
        idx = rng.choice(len(x_test), size=n, replace=False)
        cols, rows = 5, (n + 4) // 5
        fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
        for ax, i in zip(axes.ravel(), idx):
            ax.imshow(x_test[i].squeeze(), cmap="gray")
            correct = y_pred[i] == y_true[i]
            color = "green" if correct else "red"
            ax.set_title(
                f"P:{class_names[y_pred[i]]} ({proba[i].max():.0%})\nT:{class_names[y_true[i]]}",
                color=color, fontsize=10,
            )
            ax.axis("off")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Sample Predictions (green = correct, red = wrong)", fontsize=14)
        self._save(fig, "predictions.png")

    def _save(self, fig, name: str) -> None:
        path = self.outputs_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Saved figure -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the handwriting CNN.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    Evaluator(load_config(args.config)).evaluate()


if __name__ == "__main__":
    main()
