import numpy as np
import librosa
from scipy.fftpack import dct
import json

SR = 16000
N_FFT = 512
N_MELS = 40

def export_matrices():
    # 1. Mel Filterbank [40, 257]
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)

    # 2. DCT Matrix [40, 40]
    dct_matrix = dct(np.eye(N_MELS), type=2, norm='ortho')

    # 3. DFT Matrix (The New "Nuclear" Part)
    # We create the exact math matrix Numpy uses for FFT
    # Shape: [257, 512] (Real input -> Complex output)
    n = np.arange(N_FFT)
    k = np.arange(N_FFT // 2 + 1)[:, np.newaxis]
    # The formula: exp(-2j * pi * k * n / N)
    dft_matrix = np.exp(-2j * np.pi * k * n / N_FFT)

    # Split into Real and Imag parts so we can load it in JSON
    dft_real = np.real(dft_matrix)
    dft_imag = np.imag(dft_matrix)

    # 4. Save Everything
    data = {
        "mel_basis": mel_basis.tolist(),
        "dct_matrix": dct_matrix.tolist(),
        "window": np.hanning(N_FFT).tolist(),
        "dft_real": dft_real.tolist(),
        "dft_imag": dft_imag.tolist()
    }

    with open("audio_config.json", "w") as f:
        json.dump(data, f)

    print(f"✅ Exported audio_config.json (Includes DFT Matrices)")
    print(f"DFT Real Shape: {dft_real.shape}")

if __name__ == "__main__":
    export_matrices()