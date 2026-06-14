#!/usr/bin/env python3

from src import Sequential, categorical_accuracy
from src.losses import BinaryCrossentropy
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run predictions with a trained MLP model"
    )

    parser.add_argument(
        "--model", required=True, help="Path to saved model (.npz)"
    )

    parser.add_argument(
        "--scaler", required=True, help="Path to saved scaler (.npz)"
    )

    parser.add_argument(
        "--dataset", required=True, help="Path to validation dataset (.csv)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data = np.load(args.scaler)
    mean = data["mean"]
    std = data["std"]

    valid_df = pd.read_csv(args.dataset)

    # split X / y
    X_valid = valid_df.drop(columns=["label"]).to_numpy()
    y_valid = valid_df["label"].to_numpy()

    # one-hot encoding
    y_valid_ohe = np.eye(2)[y_valid]

    # standardization
    X_valid_std = (X_valid - mean) / std

    model = Sequential.load(args.model)

    print(f"[predict] running predictions on {len(X_valid_std)} samples...\n")
    y_pred = model.predict(X_valid_std, batch_size=32)

    acc = categorical_accuracy(y_valid_ohe, y_pred)

    loss_fn = BinaryCrossentropy()
    bce = loss_fn.forward(y_valid.reshape(-1, 1), y_pred[:, 1].reshape(-1, 1))

    pred_labels = np.argmax(y_pred, axis=1)
    correct = np.sum(pred_labels == y_valid)

    total = len(y_valid)

    print("[result]")
    print(f"samples              : {total}")
    print(f"correct predictions  : {correct}/{total}")
    print(f"binary cross-entropy : {bce:.6f}")
    print(f"accuracy             : {acc:.1%}")

    print("\n[per-class breakdown]")
    for c in range(2):
        mask = y_valid == c
        correct_c = np.sum(pred_labels[mask] == c)
        total_c = np.sum(mask)
        pct = correct_c / total_c if total_c > 0 else 0.0
        print(f"  class {c}  : {correct_c}/{total_c}  ({pct:.1%})")


if __name__ == "__main__":
    main()
