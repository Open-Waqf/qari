import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# Paths relative to the script
SCRIPT_DIR = Path(__file__).parent.resolve()

# 🟢 FIX: Use the raw exported graph instead of the .keras zip file
EXPORT_PATH = SCRIPT_DIR.parent / "models" / "qari_model_export"
OUTPUT_PATH = SCRIPT_DIR.parent.parent / "app" / "public" / "models" / "model_parity.json"


def generate_parity():
    print("🤖 Loading exported TensorFlow graph...")
    # tf.saved_model.load completely bypasses Keras version & config bugs
    model = tf.saved_model.load(str(EXPORT_PATH))

    np.random.seed(42)
    fake_mfcc = np.random.uniform(-3.0, 3.0, size=(1, 40, 171, 1)).astype(np.float32)

    print("🧠 Running Python prediction...")
    # Get the default inference signature
    infer = model.signatures["serving_default"]

    # Convert numpy array to TF Tensor
    input_tensor = tf.constant(fake_mfcc)

    # Run inference (grab the first output tensor from the dictionary)
    output_dict = infer(input_tensor)
    preds = list(output_dict.values())[0].numpy()[0]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "input_mfcc": fake_mfcc.flatten().tolist(),
        "shape": [1, 40, 171, 1],
        "expected_probs": preds.tolist()
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)

    print(f"✅ Parity data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_parity()
