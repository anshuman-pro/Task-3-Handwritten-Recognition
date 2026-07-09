"""
Inference API & CLI.

Loads the best saved CNN (automatically) and classifies a handwritten digit
supplied as an image path, a PIL image or a NumPy array. Returns the predicted
class, confidence and the top-k predictions.

    * as a library -> ``Predictor().predict(pil_image)``
    * as a CLI     -> ``python -m src.predict --image path/to/digit.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageOps

from src.utils import ModelError, PROJECT_ROOT, get_logger

LOGGER = get_logger("predict")

DEFAULT_MODEL = "models/best_model.keras"
IMG_SIZE = 28


class Predictor:
    """Self-contained handwritten-digit predictor with automatic model loading."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        from tensorflow.keras.models import load_model

        path = PROJECT_ROOT / (model_path or DEFAULT_MODEL)
        if not path.exists():
            raise ModelError(f"Model not found at {path}. Run `python -m src.train` first.")
        self.model = load_model(path)
        self.class_names = [str(i) for i in range(10)]
        LOGGER.info("Loaded model from %s", path)

    # ------------------------------------------------------------------ #
    def preprocess(self, image: Any) -> np.ndarray:
        """Convert an arbitrary input image to a (1, 28, 28, 1) MNIST-style tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        if not isinstance(image, Image.Image):
            raise ModelError("Unsupported image type; provide a path, PIL image or ndarray.")

        image = image.convert("L")  # grayscale
        # MNIST is white-on-black. If the background is bright, invert so strokes are white.
        if np.asarray(image).mean() > 127:
            image = ImageOps.invert(image)

        image = ImageOps.autocontrast(image)
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype="float32").reshape(1, IMG_SIZE, IMG_SIZE, 1)
        return array

    def predict(self, image: Any, top_k: int = 5) -> Dict[str, Any]:
        array = self.preprocess(image)
        proba = self.model.predict(array, verbose=0)[0]
        order = np.argsort(proba)[::-1]
        top: List[Dict[str, float]] = [
            {"class": self.class_names[i], "probability": float(proba[i])}
            for i in order[:top_k]
        ]
        best = int(order[0])
        return {
            "label": self.class_names[best],
            "confidence": float(proba[best]),
            "top_k": top,
            "probabilities": proba.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a handwritten digit image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default=None, help="Path to a model file")
    args = parser.parse_args()

    result = Predictor(args.model).predict(args.image)
    print("\n----------------------------------------")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print("  Top-5:")
    for item in result["top_k"]:
        print(f"    {item['class']}  ->  {item['probability']:.1%}")
    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
