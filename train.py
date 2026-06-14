import numpy as np
import pandas as pd
from src import Sequential, Dense, EarlyStopping

import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using a CSV dataset provided by the user."
    )

    parser.add_argument(
        "--dataset", required=True, help="Path to training dataset (.csv)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(args.dataset)

    # split X / y
    X_train = train_df.drop(columns=["label"]).to_numpy()
    y_train = train_df["label"].to_numpy()

    # one-hot encoding
    y_train_ohe = np.eye(2)[y_train]

    # standardization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train_std = (X_train - mean) / std

    # save scaler
    np.savez(
        "scaler",
        mean=mean,
        std=std,
    )

    np.random.seed(42)
    shape = X_train_std.shape

    # ---------------- THE FIRST MODLE ------------------------

    model_1 = Sequential(
        [
            Dense(24, activation="sigmoid", input_shape=shape),
            Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
            Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
            Dense(2, activation="softmax", kernel_initializer="HeUniform"),
        ]
    )

    model_1.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        _metrics=["accuracy", "precision"],
    )

    print("\n[train] starting training...\n")

    history_1 = model_1.fit(
        X_train_std,
        y_train_ohe,
        epochs=200,
        batch_size=84,
        validation_split=0.1,
        callbacks=[EarlyStopping()],
    )

    model_1.save("model_1")

    # ---------------- THE SECOND MODLE ------------------------

    model_2 = Sequential(
        [
            Dense(64, activation="sigmoid", input_shape=shape),
            Dense(64, activation="sigmoid", kernel_initializer="HeUniform"),
            Dense(64, activation="sigmoid", kernel_initializer="HeUniform"),
            Dense(2, activation="softmax", kernel_initializer="HeUniform"),
        ]
    )

    model_2.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        _metrics=["accuracy"],
    )

    print("\n[train] starting training...\n")

    history_2 = model_2.fit(
        X_train_std,
        y_train_ohe,
        epochs=200,
        batch_size=84,
        validation_split=0.1,
        callbacks=[EarlyStopping()],
    )

    model_2.save("model_2")

    # ------------------------ PLOTING ----------------------

    fig, axs = plt.subplots(2, 2, figsize=(13, 5))
    axs[0, 0].plot(history_1["accuracy"])
    axs[0, 0].plot(history_1["val_accuracy"])
    axs[0, 0].set_title("Trained Neural Network Accuracy")
    axs[0, 0].set_xlabel("epochs")
    axs[0, 0].set_ylabel("accuracy")

    axs[0, 0].legend(["training acc", "validation acc"], loc="lower right")

    axs[0, 1].plot(history_1["loss"])
    axs[0, 1].plot(history_1["val_loss"])
    axs[0, 1].set_title("Trained Neural Network Loss")
    axs[0, 1].set_xlabel("epochs")
    axs[0, 1].set_ylabel("Loss")

    axs[0, 1].legend(["training loss", "validation loss"], loc="lower right")

    axs[1, 0].plot(history_1["val_accuracy"])
    axs[1, 0].plot(history_2["val_accuracy"])
    axs[1, 0].set_title("Model_1 vs Model_2 Validation Accuracy")
    axs[1, 0].set_xlabel("epochs")
    axs[1, 0].set_ylabel("accuracy")

    axs[1, 0].legend(
        ["model_1 validation acc", "model_2 validation acc"], loc="lower right"
    )

    axs[1, 1].plot(history_1["val_loss"])
    axs[1, 1].plot(history_2["val_loss"])
    axs[1, 1].set_title("Model_1 vs Model_2 Validation Loss")
    axs[1, 1].set_xlabel("epochs")
    axs[1, 1].set_ylabel("Loss")

    axs[1, 1].legend(
        ["model_1 validation loss", "model_2 validation loss"], loc="lower right"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
