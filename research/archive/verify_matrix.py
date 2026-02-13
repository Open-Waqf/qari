import numpy as np
import librosa
import json
import scipy.io.wavfile as wav

def verify_matrix_math():
    # 1. Load the Audio Physics (JSON)
    # We read the EXACT same file the App uses
    with open("../app/public/models/audio_config.json", "r") as f:
        config = json.load(f)

    dft_real = np.array(config["dft_real"])
    dft_imag = np.array(config["dft_imag"])
    mel_basis = np.array(config["mel_basis"])
    dct_matrix = np.array(config["dct_matrix"])
    window = np.array(config["window"])

    # 2. Load the Sine Wave (Signal)
    # We manually load to ensure parity with the browser's 512 samples
    sr, y = wav.read("../app/public/test_sine.wav")
    y = y.astype(np.float32) / 32767.0 # Normalize to -1.0 to 1.0
    signal = y[0:512]

    # 3. RUN THE PIPELINE (Manual Matrix Math)

    # A. Window
    windowed = signal * window

    # B. FFT (Matrix Mult)
    # [257, 512] @ [512] = [257]
    real_part = dft_real @ windowed
    imag_part = dft_imag @ windowed
    magnitude = np.sqrt(real_part**2 + imag_part**2)

    # C. Mel Filter
    # [40, 257] @ [257] = [40]
    mel_energies = mel_basis @ magnitude

    # D. Log
    log_mel = np.log(mel_energies + 1e-6)

    # E. DCT
    # [40, 40] @ [40] = [40]
    mfccs = dct_matrix @ log_mel

    print("✅ Python Matrix Results:")
    print(mfccs[:5])

    # DEBUG: Print Spectrum to compare with App
    print("\n🔍 Spectrum Magnitude (First 5):")
    print(magnitude[:5])

if __name__ == "__main__":
    verify_matrix_math()