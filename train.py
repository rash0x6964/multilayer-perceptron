import numpy as np
import pandas as pd
from src import Sequential, Dense, EarlyStopping

from matplotlib import figure
import matplotlib.pyplot as plt


train_df = pd.read_csv("data_training.csv")

# split X / y
X_train = train_df.drop(columns=["label"]).to_numpy()
y_train = train_df["label"].to_numpy()

# one-hot encoding
y_train_ohe = np.eye(2)[y_train]

# standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train_std = ((X_train - mean) / std)


# save scaler
np.savez(
    "scaler",
    mean=mean,
    std=std,
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


fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(history['accuracy'])
axs[0].plot(history['val_accuracy'])
axs[0].set_title('Trained Neural Network Accuracy')
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('accuracy')

axs[0].legend(['training acc', 'validation acc'], loc='lower right')


axs[1].plot(history['loss'])
axs[1].plot(history['val_loss'])
axs[1].set_title('Trained Neural Network Loss')
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('Loss')

axs[1].legend(['training loss', 'validation loss'], loc='lower right')

plt.tight_layout()
plt.show()
