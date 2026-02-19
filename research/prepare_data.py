import hashlib
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Gain

# ==========================================
# ⚡ CORE SETTINGS (Synced with App Parity)
# ==========================================
SR = 22050  # 🟢 CHANGED: Must match App (was 16000)
DURATION = 2.0  # 🟢 CHANGED: Must match App (was 3.0)
SAMPLES_PER_CHUNK = int(SR * DURATION)  # 44100 samples
DATA_PATH = "datasets/audio"
OUTPUT_PATH = "models/features.npz"
APP_MODELS_DIR = Path("../app/public/models")
RECITERS_MAP_PATH = APP_MODELS_DIR / "reciters_map.json"

FRAME_LENGTH = 512
HOP_LENGTH = 256
EXPECTED_FRAMES = 1 + (SAMPLES_PER_CHUNK - FRAME_LENGTH) // HOP_LENGTH

RMS_MIN_RECITER = 0.01
RMS_MIN_BG = 0.003

TARGET_RMS = 0.10
MIN_GAIN = 0.6
# 🟢 FIX: Aligned to 5.0 to match the frontend VAD gain logic
MAX_GAIN = 5.0
RMS_FLOOR = 0.002

# -----------------------------
# SAFE CAPPING (deterministic)
# -----------------------------
# 🟢 CHANGED: Set a default cap to stop "Bullies" automatically
CAP_MINUTES_DEFAULT = 15.0
CAP_MINUTES_BY_CLASS = {
    "_background": 20.0
}

CAP_SEED = 42

# Make dataset generation as deterministic as practical.
# (Note: some audio decoding/augmentation operations can still vary across platforms.)
random.seed(CAP_SEED)
np.random.seed(CAP_SEED)

# --- 1. DEFINE THE "BAD MIC" SIMULATOR ---
augment = Compose([
    # Gain: Moderate range (-6 to +3) to avoid extreme quietness that confuses the model
    Gain(min_gain_db=-6.0, max_gain_db=3.0, p=0.8),

    # Noise: Subtle background hiss (Laptop fan / AC)
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),

    # HighPass: Cuts "Mud" and "Rumble" (80-300Hz)
    HighPassFilter(min_cutoff_freq=80, max_cutoff_freq=300, p=0.5),

    # LowPass: Cuts "Hiss" but KEEPS Voice Clarity (6000Hz+)
    LowPassFilter(min_cutoff_freq=6000, max_cutoff_freq=7800, p=0.3),
])


def normalize_signal(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    # 1. Floor Gate (Critical: Don't boost background noise)
    if r < RMS_FLOOR:
        return x

    # 2. Linear Gain
    g = TARGET_RMS / max(1e-12, r)
    g = float(np.clip(g, MIN_GAIN, MAX_GAIN))

    if abs(g - 1.0) < 1e-3:
        return x

    # 🟢 PARITY FIX: Linear Gain + Hard Clip
    y = x * g
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32, copy=False)


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
    # Robust path finding
    config_path = "../app/public/models/audio_config.json"
    if not os.path.exists(config_path):
        # Fallback if running from root
        config_path = "app/public/models/audio_config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "audio_config.json not found. Expected ../app/public/models/audio_config.json "
            "or app/public/models/audio_config.json. Build/run the app to generate it."
        )

    # Hash for parity/audit traceability
    with open(config_path, "rb") as f:
        raw = f.read()
    config_hash = hashlib.sha256(raw).hexdigest()

    config = json.loads(raw.decode("utf-8"))
    return {
        "dft_real": np.array(config["dft_real"], dtype=np.float32),
        "dft_imag": np.array(config["dft_imag"], dtype=np.float32),
        "mel_basis": np.array(config["mel_basis"], dtype=np.float32),
        "dct_matrix": np.array(config["dct_matrix"], dtype=np.float32),
        "window": np.array(config["window"], dtype=np.float32),
        "_config_hash": config_hash,
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


def _mfcc_image_from_chunk(chunk: np.ndarray) -> np.ndarray:
    # chunk: (44100,) approx
    frames = librosa.util.frame(chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).T
    mfccs = [extract_features_matrix(f) for f in frames]
    img = np.array(mfccs, dtype=np.float32).T

    if img.shape[1] != EXPECTED_FRAMES:
        raise ValueError(f"MFCC width mismatch: got {img.shape[1]}, expected {EXPECTED_FRAMES}. Check SR/DURATION.")

    # Guardrail (should always be true for 2.0s @ 22050)
    # 44100 / 256 hop ~= 171 frames approx.
    # We won't hard crash on shape unless it's zero,
    # but let's ensure it's (40, Time)
    if img.shape[0] != 40:
        raise ValueError(f"Bad MFCC shape: {img.shape} (expected (40, T))")

    return img


def process_dataset():
    X, y, groups, is_clean = [], [], [], []

    reciters = load_or_update_reciters_map(DATA_PATH)
    print(f"Locked class order ({len(reciters)}): {reciters}")
    label_map = {name: i for i, name in enumerate(reciters)}

    file_counter = 0
    step = SAMPLES_PER_CHUNK // 2  # 50% overlap
    hop_sec = step / SR

    def minutes_to_max_windows(minutes: float) -> int:
        return int((minutes * 60.0) / hop_sec)

    max_windows_by_class = {}
    for r in reciters:
        m = CAP_MINUTES_BY_CLASS.get(r, CAP_MINUTES_DEFAULT)
        max_windows_by_class[r] = (minutes_to_max_windows(m) if m else None)

    print("🧢 Caps (windows per class):")
    for r in reciters:
        cap = max_windows_by_class[r]
        if cap is not None:
            print(f"   - {r}: {cap} windows (~{CAP_MINUTES_BY_CLASS.get(r, CAP_MINUTES_DEFAULT)} min)")

    for reciter in reciters:
        print(f"Processing {reciter}...")
        reciter_path = os.path.join(DATA_PATH, reciter)
        if not os.path.isdir(reciter_path):
            print(f"⚠️ Missing folder for class '{reciter}'. Expected: {Path(reciter_path).resolve()}")
            continue
        files = sorted(f for f in os.listdir(reciter_path) if not f.startswith("."))

        # Deterministic shuffle so caps don't always take only the first sorted files
        rng = np.random.default_rng(CAP_SEED + label_map[reciter])
        files = list(files)
        rng.shuffle(files)
        cap_windows = max_windows_by_class.get(reciter)
        kept_windows = 0  # counts CLEAN windows (dirty is paired)

        for file in files:
            if not file.lower().endswith((".mp3", ".wav")):
                continue

            # ✅ If we already reached cap, stop processing more files
            if cap_windows is not None and kept_windows >= cap_windows:
                break

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
                # ✅ Stop as soon as cap reached (before doing any work)
                if cap_windows is not None and kept_windows >= cap_windows:
                    break

                chunk = audio[start:start + SAMPLES_PER_CHUNK]
                if chunk.shape[0] != SAMPLES_PER_CHUNK:
                    continue

                # RMS Check (unchanged)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                rms_min = RMS_MIN_BG if reciter == "_background" else RMS_MIN_RECITER
                if rms < rms_min:
                    continue

                try:
                    # 🟢 CALL HERE: Normalize BEFORE augmentation/features
                    norm_chunk = normalize_signal(chunk)

                    # Hard guardrail: no NaNs/Infs
                    if not np.isfinite(norm_chunk).all():
                        raise ValueError("Non-finite values in normalized chunk")

                    # 1. CLEAN (Use normalized)
                    clean_entry = _mfcc_image_from_chunk(norm_chunk)

                    # 2. DIRTY (Augment the normalized chunk)
                    dirty_chunk = augment(samples=norm_chunk, sample_rate=SR).astype(np.float32, copy=False)

                    if len(dirty_chunk) > SAMPLES_PER_CHUNK:
                        dirty_chunk = dirty_chunk[:SAMPLES_PER_CHUNK]
                    elif len(dirty_chunk) < SAMPLES_PER_CHUNK:
                        dirty_chunk = np.pad(
                            dirty_chunk, (0, SAMPLES_PER_CHUNK - len(dirty_chunk))
                        ).astype(np.float32, copy=False)

                    # Keep dirty chunk in a realistic waveform range
                    dirty_chunk = np.clip(dirty_chunk, -1.0, 1.0).astype(np.float32, copy=False)

                    dirty_entry = _mfcc_image_from_chunk(dirty_chunk)

                    # Append only if both succeeded (unchanged behavior)
                    # 1. CLEAN (Weight 1.0)
                    X.append(clean_entry)
                    y.append(label_map[reciter])
                    groups.append(file_counter)
                    is_clean.append(True)

                    # 2. DIRTY (Weight 0.4)
                    X.append(dirty_entry)
                    y.append(label_map[reciter])
                    groups.append(file_counter)
                    is_clean.append(False)

                    # ✅ IMPORTANT: count 1 “window” per CLEAN+DIRTY pair
                    kept_windows += 1

                except Exception as e:
                    print(f"⚠️ Augmentation failed, skipping chunk pair: {e}")
                    continue

            file_counter += 1

        if cap_windows is not None:
            print(
                f"🧢 {reciter}: kept {kept_windows}/{cap_windows} windows (clean), total samples added={kept_windows * 2}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)
    is_clean = np.array(is_clean, dtype=bool)

    # 🟢 ADDED: Expand dims for CNN (Height, Width, 1)
    X = X[..., np.newaxis]

    # Pairing sanity check for training normalization logic (X[::2] clean)
    if X.shape[0] % 2 != 0:
        raise RuntimeError(f"Expected even number of samples (clean/dirty pairs). Got {X.shape[0]}.")

    if X.shape[0] == 0:
        raise RuntimeError("No samples were generated. Check dataset paths and RMS thresholds.")

    # Pairing invariants: y/groups must match per (clean,dirty) pair
    if not (np.all(y[0::2] == y[1::2]) and np.all(groups[0::2] == groups[1::2])):
        raise RuntimeError("Clean/dirty pairing invariant broken (y/groups mismatch)")

    if not (np.all(is_clean[0::2]) and np.all(~is_clean[1::2])):
        raise RuntimeError("is_clean mask invariant broken (expected True/False alternating)")

    print(f"✅ Dataset Ready. Shape: {X.shape} (Includes Clean + Augmented)")
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
    np.savez(OUTPUT_PATH, X=X, y=y, groups=groups, is_clean=is_clean, mapping=label_map)
    print(f"Saved to {OUTPUT_PATH}")

    # Write a small metadata file for parity/audit traceability
    meta = {
        "sr": SR,
        "duration_sec": DURATION,
        "samples_per_chunk": SAMPLES_PER_CHUNK,
        "frame_length": FRAME_LENGTH,
        "hop_length": HOP_LENGTH,
        "expected_frames": EXPECTED_FRAMES,
        "rms_min_reciter": RMS_MIN_RECITER,
        "rms_min_bg": RMS_MIN_BG,
        "target_rms": TARGET_RMS,
        "min_gain": MIN_GAIN,
        "max_gain": MAX_GAIN,
        "rms_floor": RMS_FLOOR,
        "cap_minutes_default": CAP_MINUTES_DEFAULT,
        "cap_minutes_by_class": CAP_MINUTES_BY_CLASS,
        "cap_seed": CAP_SEED,
        "audio_config_sha256": MATRICES.get("_config_hash"),
        "num_samples": int(X.shape[0]),
        "num_classes": int(len(label_map)),
    }
    meta_path = Path(OUTPUT_PATH).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"🧾 Wrote dataset meta: {meta_path}")


if __name__ == "__main__":
    process_dataset()
