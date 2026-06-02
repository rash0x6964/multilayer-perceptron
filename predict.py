from src import Sequential, categorical_accuracy
import numpy as np
import pandas as pd

data = np.load("scaler.npz")
mean = data["mean"]
std = data["std"]

valid_df = pd.read_csv("data_validation.csv")

# split X / y
X_valid = valid_df.drop(columns=["label"])
y_valid = valid_df["label"]

# one-hot encoding
y_valid_ohe = np.eye(2)[y_valid]

# standardization
X_valid_std = ((X_valid - mean) / std).to_numpy()


model = Sequential.load("model.npy")


print(f"[predict] running predictions on {len(X_valid_std)} samples...\n")
y_pred = model.predict(X_valid_std, batch_size=32)

acc = categorical_accuracy(y_valid_ohe, y_pred)
print(f"\n[result] accuracy on {len(X_valid_std)} samples: {acc:.1%}")
