import numpy as np

# 🩹 MONKEY PATCH: Fix for NumPy 1.24+ compatibility with TFJS
# This restores deleted aliases that tensorflowjs expects
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras


# Define the Custom Layer (Must match train.py exactly)
@keras.utils.register_keras_serializable()
class SpecAugment(keras.layers.Layer):
    def __init__(self, freq_mask_param=3, time_mask_param=12, **kwargs):
        super(SpecAugment, self).__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, inputs, training=None):
        if not training:
            return inputs
        return inputs

    def get_config(self):
        config = super(SpecAugment, self).get_config()
        config.update({"freq_mask_param": self.freq_mask_param, "time_mask_param": self.time_mask_param})
        return config


def export():
    # Paths
    MODEL_PATH = "models/qari_model.keras"
    OUTPUT_DIR = "../app/public/models"  # Direct export to App

    print(f"📂 Loading model from {MODEL_PATH}...")

    # Load model with custom object scope
    with keras.utils.custom_object_scope({'SpecAugment': SpecAugment}):
        model = tf.keras.models.load_model(MODEL_PATH)

    print(f"✅ Model loaded. Input Shape: {model.input_shape}")

    # Validate Shape (Must be 171 for your current setup of 22050Hz / 2.0s)
    # Note: Keras shape might be (None, 40, 171, 1), so index 2 is width
    if model.input_shape[2] != 171:
        print(f"⚠️ WARNING: Model expects width {model.input_shape[2]}, but your data audit said 171.")
    else:
        print("✅ Shape Check: Width 171 (Matches your Data Audit)")

    # Convert to TFJS
    print(f"🚀 Converting to TensorFlow.js format in {OUTPUT_DIR}...")

    # Use the converter
    tfjs.converters.save_keras_model(model, OUTPUT_DIR)

    print("✨ Export Complete!")
    print("   1. Check '../app/public/models/' for 'model.json' and 'group1-shard1of1.bin'.")
    print("   2. Verify 'reciters_map.json' is also there.")
    print("   3. Restart your App dev server now.")


if __name__ == "__main__":
    export()
