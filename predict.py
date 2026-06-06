#!/usr/bin/env python3

from src import Sequential, categorical_accuracy
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

    # TODO: maybe rename the categorical_accuracy to Binary Cross Entropy
    # and make sure it the same equation they provied in the subject... 
    acc = categorical_accuracy(y_valid_ohe, y_pred)
    print(f"\n[result] accuracy on {len(X_valid_std)} samples: {acc:.1%}")


if __name__ == "__main__":
    main()
