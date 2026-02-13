import sys
import os
from unittest.mock import MagicMock

# 🛑 TRICK: Mock the broken Windows library BEFORE importing tensorflowjs
# This tells Python: "If anyone asks for tensorflow_decision_forests, just give them a dummy object."
sys.modules["tensorflow_decision_forests"] = MagicMock()
sys.modules["tensorflow_decision_forests.keras"] = MagicMock()

import tensorflowjs as tfjs
from keras.models import load_model

MODEL_PATH = "qari_model.h5"
OUTPUT_DIR = "../app/public/models/tfjs_model"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"⏳ Loading {MODEL_PATH}...")
# Load the model
model = load_model(MODEL_PATH)

print("🔄 Converting to TensorFlow.js (Bypassing Decision Forests)...")
# Convert
tfjs.converters.save_keras_model(model, OUTPUT_DIR)

print(f"✅ Success! Model saved to: {os.path.abspath(OUTPUT_DIR)}")