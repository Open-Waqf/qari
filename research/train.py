import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

# Settings (must match prepare_data output)
INPUT_SHAPE = (40, 186, 1)
BATCH_SIZE = 64
EPOCHS = 60


# --- SpecAugment (Gentle) ---
class SpecAugment(keras.layers.Layer):
    def __init__(self, freq_mask_param=3, time_mask_param=12, **kwargs):
        super(SpecAugment, self).__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # Frequency masking (gentle)
        num_freq = tf.shape(inputs)[1]
        f = tf.random.uniform([], minval=0, maxval=self.freq_mask_param + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, num_freq - f), dtype=tf.int32)
        mask_freq = tf.concat(
            [
                tf.ones([1, f0, 1, 1]),
                tf.zeros([1, f, 1, 1]),
                tf.ones([1, tf.maximum(0, num_freq - f0 - f), 1, 1]),
            ],
            axis=1,
        )
        inputs = inputs * mask_freq

        # Time masking (gentle)
        num_time = tf.shape(inputs)[2]
        t = tf.random.uniform([], minval=0, maxval=self.time_mask_param + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, num_time - t), dtype=tf.int32)
        mask_time = tf.concat(
            [
                tf.ones([1, 1, t0, 1]),
                tf.zeros([1, 1, t, 1]),
                tf.ones([1, 1, tf.maximum(0, num_time - t0 - t), 1]),
            ],
            axis=2,
        )
        return inputs * mask_time

    def get_config(self):
        config = super(SpecAugment, self).get_config()
        config.update(
            {"freq_mask_param": self.freq_mask_param, "time_mask_param": self.time_mask_param}
        )
        return config


def train_model():
    print("⏳ Loading dataset...")
    data = np.load("features.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    mapping = data["mapping"].item()

    # 1) Export Reciters Map
    model_output_dir = "../app/public/models"
    os.makedirs(model_output_dir, exist_ok=True)

    labels_array = [None] * len(mapping)
    for name, idx in mapping.items():
        labels_array[idx] = name
    with open(f"{model_output_dir}/reciters_map.json", "w") as f:
        json.dump(labels_array, f)

    # 2) Reshape
    X = X[..., np.newaxis]

    # 2.5) Sample weights (Clean=1.0, Dirty=0.4)
    # prepare_data saved: [clean, dirty, clean, dirty...]
    sample_weights = np.ones(len(y), dtype=np.float32)
    sample_weights[1::2] = 0.4

    # 3) Clean-only normalization stats (keep exactly as you had)
    print("📏 Calculating Normalization Stats (Clean Data Only)...")
    X_clean_only = X[::2]
    mean = float(np.mean(X_clean_only))
    std = float(np.std(X_clean_only))
    std = max(std, 1e-6)
    print(f"   Mean: {mean:.4f}, Std: {std:.4f}")

    X = (X - mean) / std

    with open(f"{model_output_dir}/normalization.json", "w") as f:
        json.dump({"mean": mean, "std": std}, f)

    # 4) Split (split weights too)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]

    print(f"✅ Data Ready. Train: {len(X_train)}, Test: {len(X_test)}")

    # 5) Model (GAP + BN + no Flatten)
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),

            # SpecAugment is optional; keep gentle because you already have dirty data
            SpecAugment(freq_mask_param=3, time_mask_param=12),

            # Block 1: Conv -> BN -> ReLU
            keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Block 3
            keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.30),

            # Head: GAP (big generalization win vs Flatten)
            keras.layers.GlobalAveragePooling2D(),

            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.40),

            keras.layers.Dense(len(mapping), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # 6) Callbacks (val_accuracy is more meaningful for you than val_loss)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    )

    print("🚀 Starting Training (GAP + Sample Weights)...")
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test, w_test),
        callbacks=[early_stop, reduce_lr],
    )

    model.save("qari_model.h5")
    print("\n✅ Training Complete.")
    print("⚠️ NOW RUN: python convert_wizard.py")


if __name__ == "__main__":
    train_model()
