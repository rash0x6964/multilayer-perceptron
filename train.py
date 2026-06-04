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
    "scaler",
    mean=mean.to_numpy(),
    std=std.to_numpy(),
)

np.random.seed(42)
shape = X_train_std.shape


model = Sequential(
    [
        Dense(24, activation="sigmoid", input_shape=shape),
        Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
        Dense(24, activation="sigmoid", kernel_initializer="HeUniform"),
        Dense(2, activation="softmax", kernel_initializer="HeUniform"),
    ]
)

model.compile(
    optimizer="sgd",
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
    callbacks=[EarlyStopping()],
)

model.save("model")
