import json
import os
from pathlib import Path

import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Gain

# Settings
SR = 16000
DURATION = 3.0
SAMPLES_PER_CHUNK = int(SR * DURATION)  # 48000 samples
DATA_PATH = "datasets/audio"
OUTPUT_PATH = "datasets/features.npz"
APP_MODELS_DIR = Path("../app/public/models")
RECITERS_MAP_PATH = APP_MODELS_DIR / "reciters_map.json"

# --- 1. DEFINE THE "BAD MIC" SIMULATOR ---
augment = Compose([
    # Gain: Moderate range (-6 to +3) to avoid extreme quietness that confuses the model
    Gain(min_gain_db=-6.0, max_gain_db=3.0, p=0.8),

    # Noise: Subtle background hiss (Laptop fan / AC)
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),

    # HighPass: Cuts "Mud" and "Rumble" (80-300Hz)
    # This prevents the "Fares Abbad" confusion caused by deep bass
    HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=300, p=0.5),

    # LowPass: Cuts "Hiss" but KEEPS Voice Clarity (6000Hz+)
    # 🛑 CRITICAL: Do not go below 6000Hz, or Ghamdi will sound muffled
    LowPassFilter(min_cutoff_freq=6000, max_cutoff_freq=7800, p=0.3),
])


def load_or_update_reciters_map(data_path: str) -> list[str]:
    # folders present on disk
    found = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and not d.startswith(".")
    )

    # load existing ordering if present (this preserves indices)
    if RECITERS_MAP_PATH.exists():
        with open(RECITERS_MAP_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)  # should be a LIST of class names
        if not isinstance(existing, list):
            raise ValueError("reciters_map.json must be a JSON list of class names in index order.")
        final = list(existing)
    else:
        existing = []
        final = []
        # fresh repo case: still put _background at the end if it exists
        if "_background" in found:
            found = [x for x in found if x != "_background"] + ["_background"]
        final = list(found)

    # append new folders at end (keeps old indices intact)
    changed = False
    for name in found:
        if name not in final:
            final.append(name)
            changed = True
            print(f"➕ New class appended: {name} -> index {len(final) - 1}")

    # warn if something in JSON missing on disk (do not reorder!)
    for name in existing:
        if name not in found:
            print(f"⚠️ In reciters_map.json but missing folder in audio/: {name}")

    # write back only if needed
    if changed or not RECITERS_MAP_PATH.exists():
        RECITERS_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RECITERS_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"💾 reciters_map.json updated: {len(final)} classes")

    return final


def load_matrices():
    with open("../app/public/models/audio_config.json", "r") as f:
        config = json.load(f)
    return {
        "dft_real": np.array(config["dft_real"], dtype=np.float32),
        "dft_imag": np.array(config["dft_imag"], dtype=np.float32),
        "mel_basis": np.array(config["mel_basis"], dtype=np.float32),
        "dct_matrix": np.array(config["dct_matrix"], dtype=np.float32),
        "window": np.array(config["window"], dtype=np.float32),
    }


MATRICES = load_matrices()


def extract_features_matrix(frame_512: np.ndarray) -> np.ndarray:
    # frame_512 is exactly 512 samples
    windowed = frame_512 * MATRICES["window"]
    real = MATRICES["dft_real"] @ windowed
    imag = MATRICES["dft_imag"] @ windowed
    mag = np.sqrt(real ** 2 + imag ** 2)
    mel = MATRICES["mel_basis"] @ mag

    # Match app epsilon
    log_mel = np.log(mel + 1e-6)
    mfcc = MATRICES["dct_matrix"] @ log_mel
    return mfcc


def _mfcc_image_from_chunk(chunk_48k: np.ndarray) -> np.ndarray:
    # chunk_48k: (48000,)
    frames = librosa.util.frame(chunk_48k, frame_length=512, hop_length=256).T  # (186, 512)
    mfccs = [extract_features_matrix(f) for f in frames]  # list of (40,)
    img = np.array(mfccs, dtype=np.float32).T  # (40, 186)

    # Guardrail (should always be true)
    if img.shape != (40, 186):
        raise ValueError(f"Bad MFCC shape: {img.shape} (expected (40, 186))")

    return img


def process_dataset():
    X, y, groups = [], [], []

    reciters = load_or_update_reciters_map(DATA_PATH)
    print(f"Locked class order ({len(reciters)}): {reciters}")
    label_map = {name: i for i, name in enumerate(reciters)}

    file_counter = 0
    step = SAMPLES_PER_CHUNK // 2  # 50% overlap

    for reciter in reciters:
        print(f"Processing {reciter}...")
        reciter_path = os.path.join(DATA_PATH, reciter)
        if not os.path.isdir(reciter_path):
            print(f"⚠️ Missing folder for class '{reciter}'. Expected: {Path(reciter_path).resolve()}")
            continue
        files = sorted(f for f in os.listdir(reciter_path) if not f.startswith("."))

        for file in files:
            if not file.lower().endswith((".mp3", ".wav")):
                continue

            file_path = os.path.join(reciter_path, file)

            try:
                # Explicit decode behavior (more stable)
                audio, _ = librosa.load(
                    file_path, sr=SR, mono=True, res_type="soxr_hq"
                )
                audio = audio.astype(np.float32, copy=False)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # ✅ FIX: include last valid start (+1)
            last_start = len(audio) - SAMPLES_PER_CHUNK
            if last_start < 0:
                file_counter += 1
                continue

            for start in range(0, last_start + 1, step):
                chunk = audio[start:start + SAMPLES_PER_CHUNK]
                if chunk.shape[0] != SAMPLES_PER_CHUNK:
                    continue

                # RMS Check (unchanged)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < 0.01:
                    continue

                try:
                    # Clean
                    clean_entry = _mfcc_image_from_chunk(chunk)

                    # Dirty (keep exact length)
                    dirty_chunk = augment(samples=chunk, sample_rate=SR).astype(np.float32, copy=False)
                    if len(dirty_chunk) > SAMPLES_PER_CHUNK:
                        dirty_chunk = dirty_chunk[:SAMPLES_PER_CHUNK]
                    elif len(dirty_chunk) < SAMPLES_PER_CHUNK:
                        dirty_chunk = np.pad(
                            dirty_chunk, (0, SAMPLES_PER_CHUNK - len(dirty_chunk))
                        ).astype(np.float32, copy=False)

                    dirty_entry = _mfcc_image_from_chunk(dirty_chunk)

                    # Append only if both succeeded (unchanged behavior)
                    # 1. CLEAN (Weight 1.0)
                    X.append(clean_entry)
                    y.append(label_map[reciter])
                    groups.append(file_counter)

                    # 2. DIRTY (Weight 0.4)
                    X.append(dirty_entry)
                    y.append(label_map[reciter])
                    groups.append(file_counter)

                except Exception as e:
                    print(f"⚠️ Augmentation failed, skipping chunk pair: {e}")
                    continue

            file_counter += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    # Pairing sanity check for training normalization logic (X[::2] clean)
    if X.shape[0] % 2 != 0:
        raise RuntimeError(f"Expected even number of samples (clean/dirty pairs). Got {X.shape[0]}.")

    print(f"✅ Dataset Ready. Shape: {X.shape} (Includes Clean + Augmented)")
    np.savez(OUTPUT_PATH, X=X, y=y, groups=groups, mapping=label_map)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_dataset()
