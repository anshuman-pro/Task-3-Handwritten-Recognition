"""
CNN architecture.

A compact but strong VGG-style convolutional network. Normalization
(``Rescaling``) and data augmentation (``RandomRotation`` / ``RandomZoom`` /
``RandomTranslation``) are baked in as the first layers, so the network accepts
raw ``[0, 255]`` images and augmentation is applied only during training.
"""

from __future__ import annotations

from src.utils import Config, get_logger

LOGGER = get_logger("model")


class CNNModel:
    """Builder for the handwritten-character recognition CNN."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.img_size = int(config.get("data.img_size", 28))
        self.num_classes = int(config.get("data.num_classes", 10))

    def _augmentation(self):
        from tensorflow.keras import Sequential, layers

        return Sequential(
            [
                layers.RandomRotation(self.config.get("augmentation.rotation", 0.06)),
                layers.RandomZoom(self.config.get("augmentation.zoom", 0.10)),
                layers.RandomTranslation(
                    self.config.get("augmentation.translation", 0.10),
                    self.config.get("augmentation.translation", 0.10),
                ),
            ],
            name="augmentation",
        )

    def build(self):
        """Construct and return an uncompiled Keras model."""
        from tensorflow.keras import Input, Model, layers

        filters = self.config.get("model.conv_filters", [32, 64, 128])
        dense_units = self.config.get("model.dense_units", 128)
        drop_conv = self.config.get("model.dropout_conv", 0.25)
        drop_dense = self.config.get("model.dropout_dense", 0.40)

        inputs = Input(shape=(self.img_size, self.img_size, 1), name="image")
        x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
        if self.config.get("augmentation.enabled", True):
            x = self._augmentation()(x)

        for i, f in enumerate(filters):
            x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}_a")(x)
            x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}_b")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.MaxPooling2D(name=f"pool{i}")(x)
            x = layers.Dropout(drop_conv, name=f"drop{i}")(x)

        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(dense_units, activation="relu", name="fc")(x)
        x = layers.BatchNormalization(name="bn_fc")(x)
        x = layers.Dropout(drop_dense, name="drop_fc")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="predictions")(x)

        model = Model(inputs, outputs, name="handwriting_cnn")
        LOGGER.info("Built CNN with %d trainable layers", len(model.trainable_variables))
        return model

    def build_and_compile(self):
        """Return a compiled model ready for ``fit``."""
        from tensorflow.keras.optimizers import Adam

        model = self.build()
        model.compile(
            optimizer=Adam(learning_rate=self.config.get("training.learning_rate", 1e-3)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
