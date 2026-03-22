import librosa
import numpy as np
import os
from pathlib import Path
import json

SR = 22050
DURATION = 2.0

REPRO_DIR = Path("../app/tests/repro")
SOURCE_DIR = REPRO_DIR / "source"

pairs = {
    "ahmed_phone_mic.wav": "ahmed_talib_hameed_113.mp3",
    "maher_phone_mic.wav": "maher_al_muaiqly_109.mp3",
    "mishary_phone_mic.wav": "mishary_al_afasy_114.mp3",
    "raad_phone_mic.wav": "raad_al_kurdi_082.mp3"
}

def get_avg_spectrum(y, sr):
    n_fft = 512
    hop_length = 256
    # Use windowed frames to match the app's extraction
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))
    return np.mean(S, axis=1)

def analyze():
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    results = {}

    for phone_file, source_file in pairs.items():
        phone_path = REPRO_DIR / phone_file
        source_path = SOURCE_DIR / source_file

        if not phone_path.exists() or not source_path.exists():
            continue

        y_phone, _ = librosa.load(phone_path, sr=SR, duration=DURATION)
        y_source, _ = librosa.load(source_path, sr=SR, duration=DURATION)

        # RMS Normalize
        y_phone = y_phone / (np.sqrt(np.mean(y_phone**2)) + 1e-9)
        y_source = y_source / (np.sqrt(np.mean(y_source**2)) + 1e-9)

        spec_phone = get_avg_spectrum(y_phone, SR)
        spec_source = get_avg_spectrum(y_source, SR)

        # Gain ratio (Phone / Source)
        gain_ratio = (spec_phone + 1e-6) / (spec_source + 1e-6)
        results[phone_file] = gain_ratio

    if not results:
        print("No pairs found.")
        return

    avg_gain = np.mean(list(results.values()), axis=0)

    print("\n--- Calibration Anchors (Phone/Source Gain Ratio) ---")
    print(f"{'Freq (Hz)':>10} | {'Gain Ratio':>10}")
    print("-" * 25)
    # We want these points for our interpolation
    anchors_hz = [0, 180, 450, 1000, 2200, 3800, 5600, 9000, SR // 2]
    for hz in anchors_hz:
        idx = np.argmin(np.abs(freqs - hz))
        print(f"{freqs[idx]:10.0f} | {avg_gain[idx]:10.4f}")

    # Also print specifically for Ahmed and Maher to see if they are different
    for qari in ["ahmed", "maher", "raad", "mishary"]:
        fname = f"{qari}_phone_mic.wav"
        if fname in results:
            print(f"\n--- {qari.upper()} Gain Ratio ---")
            for hz in anchors_hz:
                idx = np.argmin(np.abs(freqs - hz))
                print(f"{freqs[idx]:10.0f} | {results[fname][idx]:10.4f}")

if __name__ == "__main__":
    analyze()
