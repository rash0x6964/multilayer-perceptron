import argparse
import numpy as np
import pandas as pd


def train_test_split(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    split_idx = int(len(df) * (1 - test_size))

    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)

    return train_df, valid_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Validation set ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    train_df, valid_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed
    )

    train_df.to_csv("data_training.csv", index=False)
    valid_df.to_csv("data_validation.csv", index=False)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")

    print("Saved:")
    print("  data_training.csv")
    print("  data_validation.csv")


if __name__ == "__main__":
    main()
