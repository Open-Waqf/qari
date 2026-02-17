import json

import librosa
import numpy as np
from scipy.fftpack import dct

# 🟢 UPDATE THESE TO MATCH NEW SETUP
SR = 22050
N_FFT = 512
N_MELS = 40


def export_matrices():
    # 1. Mel Filterbank [40, 257]
    # We use the same parameters as your Python feature extractor
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)

    # 2. DCT Matrix [40, 40]
    dct_matrix = dct(np.eye(N_MELS), type=2, norm='ortho')

    # 3. DFT Matrix (The "Nuclear" Option)
    # Allows the App to do the exact same FFT as Python
    n = np.arange(N_FFT)
    k = np.arange(N_FFT // 2 + 1)[:, np.newaxis]
    dft_matrix = np.exp(-2j * np.pi * k * n / N_FFT)

    dft_real = np.real(dft_matrix)
    dft_imag = np.imag(dft_matrix)

    # 4. Save
    data = {
        "mel_basis": mel_basis.tolist(),
        "dct_matrix": dct_matrix.tolist(),
        "window": np.hanning(N_FFT).tolist(),
        "dft_real": dft_real.tolist(),
        "dft_imag": dft_imag.tolist()
    }

    output_path = "../app/public/models/audio_config.json"
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"✅ Exported EAR CONFIG to {output_path}")
    print(f"   SR={SR}, Mels={N_MELS}, FFT={N_FFT}")


if __name__ == "__main__":
    export_matrices()
