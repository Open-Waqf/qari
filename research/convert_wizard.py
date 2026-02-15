import os
import sys
from unittest.mock import MagicMock

import tensorflow as tf

# 🛑 Mock TFDF BEFORE importing tensorflowjs
sys.modules["tensorflow_decision_forests"] = MagicMock()
sys.modules["tensorflow_decision_forests.keras"] = MagicMock()

import tensorflowjs as tfjs


class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask_param=5, time_mask_param=10, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, inputs, training=None):
        # Identity for export/inference
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
        })
        return cfg


MODEL_PATH = "qari_model.h5"
OUTPUT_DIR = "../app/public/models/tfjs_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"⏳ Loading {MODEL_PATH} with custom SpecAugment layer...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"SpecAugment": SpecAugment},
    compile=False,  # ✅ important: no training state needed
)

print("🔄 Converting to TensorFlow.js...")
tfjs.converters.save_keras_model(model, OUTPUT_DIR)

print(f"✅ Success! Model saved to: {os.path.abspath(OUTPUT_DIR)}")
