import json
import os

import librosa
import numpy as np

# Settings
SR = 16000
DURATION = 3.0  # We train on 3-second chunks
SAMPLES_PER_CHUNK = int(SR * DURATION)
DATA_PATH = "audio/"  # Point this to your manhuw audio folder
OUTPUT_PATH = "features.npz"


def load_matrices():
    # Load the verified physics
    with open("../app/public/models/audio_config.json", "r") as f:
        config = json.load(f)
    return {
        "dft_real": np.array(config["dft_real"]),
        "dft_imag": np.array(config["dft_imag"]),
        "mel_basis": np.array(config["mel_basis"]),
        "dct_matrix": np.array(config["dct_matrix"]),
        "window": np.array(config["window"])
    }


MATRICES = load_matrices()


def extract_features_matrix(signal):
    # This function mimics the App's "CustomAudioExtractor" exactly

    # 1. Window
    windowed = signal * MATRICES["window"]

    # 2. FFT
    real = MATRICES["dft_real"] @ windowed
    imag = MATRICES["dft_imag"] @ windowed
    mag = np.sqrt(real ** 2 + imag ** 2)

    # 3. Mel
    mel = MATRICES["mel_basis"] @ mag

    # 4. Log
    log_mel = np.log(mel + 1e-6)

    # 5. DCT
    mfcc = MATRICES["dct_matrix"] @ log_mel
    return mfcc


def process_dataset():
    X = []  # Features
    y = []  # Labels (Reciter IDs)
    groups = []

    reciters = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"Found {len(reciters)} reciters: {reciters}")

    label_map = {name: i for i, name in enumerate(reciters)}

    file_counter = 0
    for reciter in reciters:
        print(f"Processing {reciter}...")
        reciter_path = os.path.join(DATA_PATH, reciter)
        files = sorted(os.listdir(reciter_path))

        for file in files:
            if not file.endswith((".mp3", ".wav")): continue

            # Load Audio
            file_path = os.path.join(reciter_path, file)
            try:
                audio, _ = librosa.load(file_path, sr=SR)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Split into 3-second chunks (with 50% overlap)
            step = SAMPLES_PER_CHUNK // 2
            for start in range(0, len(audio) - SAMPLES_PER_CHUNK, step):
                chunk = audio[start: start + SAMPLES_PER_CHUNK]

                # 1. NEW: Calculate Volume (RMS)
                rms = np.sqrt(np.mean(chunk ** 2))

                # 2. Skip if too quiet (Threshold ~0.01 is usually good for normalized audio)
                if rms < 0.01:
                    continue

                # We need to process this chunk frame-by-frame
                # The App processes 512 samples at a time
                # 3 seconds * 16000 = 48000 samples
                # 48000 / 256 (hop) = ~187 frames

                # Standard Librosa sliding window logic, but using OUR matrix math
                # To speed up training, we can use a trick:
                # We know our matrix math is linear. We can use librosa's frame utility
                frames = librosa.util.frame(chunk, frame_length=512, hop_length=256).T

                # Apply our Matrix Math to every frame
                # (We do this in a loop or vectorized. Loop is clearer for now)
                chunk_mfccs = []
                for frame in frames:
                    mfcc = extract_features_matrix(frame)
                    chunk_mfccs.append(mfcc)

                chunk_mfccs = np.array(chunk_mfccs)  # Shape: [Time, 40]

                # Transpose to [40, Time] to match standard Image shape (Height, Width)
                X.append(chunk_mfccs.T)
                y.append(label_map[reciter])
                groups.append(file_counter)

            file_counter += 1

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print(f"✅ Dataset Ready. Shape: {X.shape}")
    np.savez(OUTPUT_PATH, X=X, y=y, groups=groups, mapping=label_map)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_dataset()
