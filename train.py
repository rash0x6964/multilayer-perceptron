import numpy as np
import pandas as pd
from src import Sequential, Dense, EarlyStopping


train_df = pd.read_csv("data_training.csv")

# split X / y
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

# one-hot encoding
y_train_ohe = np.eye(2)[y_train]

# standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train_std = ((X_train - mean) / std).to_numpy()


# save scaler
np.savez(
    "scaler.npy",
    mean=mean.to_numpy(),
    std=std.to_numpy(),
)


shape = X_train_std.shape

model = Sequential([
    Dense(24, activation="sigmoid", input_shape=shape),
    Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
    Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
    Dense(2,  activation="softmax", kernel_initializer="HeUniform"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n[train] starting training...\n")

history = model.fit(
    X_train_std,
    y_train_ohe,
    epochs=200,
    batch_size=84,
    validation_split=0.1,
    callbacks=[EarlyStopping()]
)

n_epochs = len(history["loss"])
final_loss = history["loss"][-1]
final_acc = history["accuracy"][-1]
best_val = min(history["val_loss"])

print(f"\n[result] trained for {n_epochs} epochs")
print(f"[result] final train loss : {final_loss:.4f}")
print(f"[result] final train acc  : {final_acc:.4f}")
print(f"[result] best val loss    : {best_val:.4f}")

model.save("model")
print("\n[save] model -> model.npy")
print("[save] scaler -> scaler.npy")
