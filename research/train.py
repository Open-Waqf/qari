import json
import os
import subprocess

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow import keras

# Settings
INPUT_SHAPE = (40, 186, 1)
BATCH_SIZE = 32
EPOCHS = 25


def train_model():
    print("⏳ Loading dataset...")
    if not os.path.exists("features.npz"):
        print("❌ Error: features.npz not found.")
        return

    data = np.load("features.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    mapping = data["mapping"].item()  # {'afasy': 0, 'sudais': 1...}

    # Save the mapping to a JSON file (App needs this to know who is who!)
    # Ensure the app folder exists
    model_output_dir = "../app/public/models"
    os.makedirs(model_output_dir, exist_ok=True)

    # Create a sorted list based on IDs
    labels_array = [None] * len(mapping)
    for name, idx in mapping.items():
        labels_array[idx] = name

    with open(f"{model_output_dir}/reciters_map.json", "w") as f:
        json.dump(labels_array, f)  # Save as [ "afasy", "sudais", ... ]
    print(f"✅ Saved reciters_map.json to {model_output_dir}")

    # 2. Preprocessing
    # Reshape X to fit CNN: (Batch, Height, Width, Channels)
    # Our prepared data is (Batch, 40, 186), we add '1' for "Grayscale Channel"
    X = X[..., np.newaxis]

    # Normalize inputs (Standard ML best practice)
    # Our features are roughly -100 to 50. We scale to roughly 0-1
    mean = float(np.mean(X))  # Convert to standard float for JSON
    std = float(np.std(X))
    X = (X - mean) / std
    print(f"Dataset Normalized. Mean: {mean:.2f}, Std: {std:.2f}")

    # --- 1. NEW: EXPORT NORMALIZATION ---
    normalization_data = {"mean": mean, "std": std}
    with open(f"{model_output_dir}/normalization.json", "w") as f:
        json.dump(normalization_data, f)
    print(f"✅ Saved normalization.json to {model_output_dir}")

    # --- 2. NEW: LEAKAGE-PROOF SPLIT ---
    if "groups" in data:
        groups = data["groups"]
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"✅ Splitting by FILE (Leakage fixed). Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        print("⚠️ Warning: No 'groups' found. Using random split (Potential Leakage).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # -----------------------------------

    # 3. Build the Model (CNN)
    # This architecture is optimized for Spectrograms (Audio Images)
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),

        # Layer 1: Detects basic edges/textures
        keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.BatchNormalization(),

        # Layer 2: Detects shapes
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Layer 3: Detects complex voice patterns
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),

        # Flatten and Classify
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),

        # Output Layer: One neuron per Reciter
        keras.layers.Dense(len(mapping), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Train
    print("🚀 Starting Training...")
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test))

    # 5. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")

    # 6. Save as Keras Model first
    model.save("qari_model.h5")
    print("💾 Saved Keras model to qari_model.h5")

    # 7. Convert to TFJS (The Final Artifact)
    print("🔄 Converting to TensorFlow.js...")

    # We use 'subprocess' to call the converter command line tool
    output_path = f"{model_output_dir}/tfjs_model"
    cmd = [
        "tensorflowjs_converter",
        "--input_format=keras",
        "qari_model.h5",
        output_path
    ]
    # Run the command
    subprocess.run(cmd, check=True)

    if os.path.exists(output_path):
        print(f"🎉 SUCCESS! Model Converted! Saved to {output_path}")
    else:
        print("⚠️ Warning: Conversion might have failed. Check if 'tensorflowjs_converter' is in your PATH.")


if __name__ == "__main__":
    train_model()
