import src
from src import layers
import numpy as np


if __name__ == "__main__":

    np.random.seed(42)

    input = np.random.random((32, 30))
    input_shape = (input.shape[-1],)
    output_shape = (2,)

    model = src.Sequential(
        [
            layers.Dense(
                24,
                activation="sigmoid",
                kernel_initializer="heUniform",
                input_shape=input_shape,
            ),
            layers.Dense(24, activation="sigmoid",
                         kernel_initializer="heUniform"),
            layers.Dense(24, activation="sigmoid",
                         kernel_initializer="heUniform"),
            layers.Dense(
                output_shape, activation="softmax", kernel_initializer="heUniform"
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",  # categoricalCrossentropy
        metrics=["accuracy"],
    )

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    # >>> history_dict = history.history
    # >>> history_dict.keys()
    # dict_keys(["accuracy", "loss", "val_accuracy", "val_loss"])
