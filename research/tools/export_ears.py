import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.fftpack import dct

# 🟢 UPDATE THESE TO MATCH NEW SETUP
SR = 22050
N_FFT = 512
N_MELS = 40

# --- PATH SETUP ---
# Resolves paths dynamically so you can run the script from anywhere
SCRIPT_DIR = Path(__file__).parent.resolve()
RESEARCH_MODELS_DIR = SCRIPT_DIR.parent / "models"
APP_MODELS_DIR = SCRIPT_DIR.parent.parent / "app" / "public" / "models"

JSON_PATH = RESEARCH_MODELS_DIR / "audio_config.json"
BIN_PATH = APP_MODELS_DIR / "audio_config.bin"


def save_to_bin(mel_basis, dct_matrix, window, dft_real, dft_imag):
    """Concatenates the matrices and writes them to a raw Float32 binary file."""
    binary_data = np.concatenate([
        np.array(mel_basis).flatten(),
        np.array(dct_matrix).flatten(),
        np.array(window).flatten(),
        np.array(dft_real).flatten(),
        np.array(dft_imag).flatten()
    ]).astype(np.float32)

    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BIN_PATH, "wb") as f:
        f.write(binary_data.tobytes())

    print(f"✅ Exported BINARY to {BIN_PATH}")
    print(f"   Size: {len(binary_data) * 4 / 1024 / 1024:.2f} MB")


def export_matrices():
    """Calculates matrices using librosa/scipy, saves JSON locally, and exports BIN to app."""
    print(f"🧮 Computing audio matrices (SR={SR}, Mels={N_MELS}, FFT={N_FFT})...")

    # 1. Mel Filterbank [40, 257]
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)

    # 2. DCT Matrix [40, 40]
    dct_matrix = dct(np.eye(N_MELS), type=2, norm='ortho')

    # 3. DFT Matrix
    n = np.arange(N_FFT)
    k = np.arange(N_FFT // 2 + 1)[:, np.newaxis]
    dft_matrix = np.exp(-2j * np.pi * k * n / N_FFT)

    dft_real = np.real(dft_matrix)
    dft_imag = np.imag(dft_matrix)

    # 4. Window [512]
    window = np.hanning(N_FFT)

    # 5. Save JSON to research/models/
    RESEARCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "mel_basis": mel_basis.tolist(),
        "dct_matrix": dct_matrix.tolist(),
        "window": window.tolist(),
        "dft_real": dft_real.tolist(),
        "dft_imag": dft_imag.tolist()
    }

    with open(JSON_PATH, "w") as f:
        json.dump(data, f)
    print(f"✅ Exported JSON to {JSON_PATH}")

    # 6. Save BIN to app/public/models/
    save_to_bin(mel_basis, dct_matrix, window, dft_real, dft_imag)


def convert_only():
    """Reads existing JSON from research and converts it directly to BIN for the app."""
    if not JSON_PATH.exists():
        print(f"❌ Cannot convert: {JSON_PATH} does not exist.")
        print("   Run the script normally (without --convert-only) to generate it first.")
        return

    print(f"🔄 Reading existing JSON from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Convert the JSON arrays directly to the bin file
    save_to_bin(
        data["mel_basis"],
        data["dct_matrix"],
        data["window"],
        data["dft_real"],
        data["dft_imag"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Audio Matrices for Qari Finder")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert the existing JSON in research/models to BIN in app/public/models"
    )

    args = parser.parse_args()

    if args.convert_only:
        convert_only()
    else:
        export_matrices()
