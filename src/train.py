# Imports


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide all WARNING/ERROR from TF
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Pretend no GPU exists
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import image_dataset_from_directory


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


set_seed(31415)

# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=18,
    titlepad=10,
)
plt.rc("image", cmap="magma")
warnings.filterwarnings("ignore")  # to clean up output cells

DIR = "./input/leaves"
CLASS_NAMES = sorted(os.listdir(DIR))
with open("class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f)

# Load training and validation sets
ds_valid_ = image_dataset_from_directory(
    "./input/leaves",
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123,
)
ds_train_ = image_dataset_from_directory(
    "./input/leaves",
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123,
)


# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(
        convert_to_float
    ).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(
        convert_to_float
    ).cache().prefetch(buffer_size=AUTOTUNE)


model = keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(epsilon=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=100,
    verbose=0,
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()

model.save("my_model.keras")
