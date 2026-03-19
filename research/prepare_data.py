import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
from audiomentations import (
    Compose,
    OneOf,
    AddGaussianNoise,
    AddGaussianSNR,
    HighPassFilter,
    LowPassFilter,
    Gain,
    BandPassFilter,
    AirAbsorption,
    BitCrush,
    ClippingDistortion,
    Mp3Compression,
    TanhDistortion,
)

sys.stdout.reconfigure(line_buffering=True)

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
CAP_MINUTES_DEFAULT = float(os.environ.get("QARI_CAP_MINUTES_DEFAULT", "15.0"))
CAP_MINUTES_BY_CLASS = {
    "_background": float(os.environ.get("QARI_CAP_MINUTES_BACKGROUND", "20.0"))
}

CAP_SEED = 42

# 🟢 NEW: Per-file cap (prevents single long recording dominating val/test)
CAP_MINUTES_PER_FILE_DEFAULT = float(os.environ.get("QARI_CAP_MINUTES_PER_FILE_DEFAULT", "4.0"))
CAP_MINUTES_PER_FILE_BY_CLASS = {
    "_background": float(os.environ.get("QARI_CAP_MINUTES_PER_FILE_BACKGROUND", "1.0"))
}

# Make dataset generation as deterministic as practical.
# (Note: some audio decoding/augmentation operations can still vary across platforms.)
random.seed(CAP_SEED)
np.random.seed(CAP_SEED)

# --- 1. DEFINE THE "BAD MIC" SIMULATOR ---
augment = Compose([
    # Gain drift between playback volume and browser mic AGC behavior.
    Gain(min_gain_db=-10.0, max_gain_db=4.0, p=0.85),

    # Real speaker-to-mic captures often include either hiss or lower-SNR room noise.
    OneOf([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.007, p=1.0),
        AddGaussianSNR(min_snr_db=8.0, max_snr_db=28.0, p=1.0),
    ], p=0.45),

    # Simulate phone speaker / laptop mic band-limiting.
    OneOf([
        BandPassFilter(min_center_freq=650.0, max_center_freq=2400.0, min_bandwidth_fraction=0.6,
                       max_bandwidth_fraction=1.6, p=1.0),
        HighPassFilter(min_cutoff_freq=120, max_cutoff_freq=380, p=1.0),
        LowPassFilter(min_cutoff_freq=2800, max_cutoff_freq=6800, p=1.0),
    ], p=0.65),

    # Speaker-to-air-to-mic high-frequency loss and cheap playback artifacts.
    AirAbsorption(min_temperature=10.0, max_temperature=20.0, min_humidity=30.0, max_humidity=80.0, p=0.20),
    BitCrush(min_bit_depth=6, max_bit_depth=12, p=0.12),

    # Cheap-speaker / browser / messaging compression artifacts.
    Mp3Compression(min_bitrate=24, max_bitrate=96, backend="pydub", p=0.35),

    # Saturation from phone speaker or hot mic path.
    OneOf([
        ClippingDistortion(min_percentile_threshold=2, max_percentile_threshold=10, p=1.0),
        TanhDistortion(min_distortion=0.02, max_distortion=0.18, p=1.0),
    ], p=0.20),
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


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    config_path = "models/audio_config.json"
    if not os.path.exists(config_path):
        # Fallback if running from root
        config_path = "research/models/audio_config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "audio_config.json not found. Expected models/audio_config.json "
            "or research/models/audio_config.json. Build/run the app to generate it."
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
print("audio_config hash:", MATRICES.get("_config_hash"))
print("dft_real:", MATRICES["dft_real"].shape)
print("mel_basis:", MATRICES["mel_basis"].shape)
print("dct_matrix:", MATRICES["dct_matrix"].shape)
print("window:", MATRICES["window"].shape)


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
    group_meta = {}  # group_id -> {class,file,rel_path,sha256}

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
        # Per-file cap (in clean windows) to avoid one long file dominating.
        # Also compute a fair-share budget per file so reduced-cap runs do not
        # collapse to just the first few shuffled files for large classes.
        file_cap_minutes = CAP_MINUTES_PER_FILE_BY_CLASS.get(reciter, CAP_MINUTES_PER_FILE_DEFAULT)
        file_cap_windows = minutes_to_max_windows(file_cap_minutes) if file_cap_minutes else None
        fair_share_by_file = {}
        if cap_windows is not None and files:
            base = cap_windows // len(files)
            remainder = cap_windows % len(files)
            for idx, fname in enumerate(files):
                fair_share = base + (1 if idx < remainder else 0)
                fair_share_by_file[fname] = max(1, fair_share)
        kept_windows = 0  # counts CLEAN windows (dirty is paired)
        t_decode = t_feat = t_aug = 0.0

        for file in files:
            if not file.lower().endswith((".mp3", ".wav")):
                continue

            # ✅ If we already reached cap, stop processing more files
            if cap_windows is not None and kept_windows >= cap_windows:
                break

            file_path = os.path.join(reciter_path, file)

            kept_windows_in_file = 0
            per_file_limit = file_cap_windows
            fair_share_limit = fair_share_by_file.get(file)
            if fair_share_limit is not None:
                per_file_limit = fair_share_limit if per_file_limit is None else min(per_file_limit, fair_share_limit)

            # Save group metadata for debugging / disjointness checks
            rel_path = f"{reciter}/{file}"
            try:
                group_meta[int(file_counter)] = {
                    "class": reciter,
                    "file": file,
                    "rel_path": rel_path,
                    "sha256": sha256_file(file_path),
                }
            except Exception as _e:
                group_meta[int(file_counter)] = {
                    "class": reciter,
                    "file": file,
                    "rel_path": rel_path,
                    "sha256": None,
                }

            try:
                # Explicit decode behavior (more stable)
                t0 = time.perf_counter()
                audio, _ = librosa.load(
                    file_path, sr=SR, mono=True, res_type="soxr_hq"
                )
                t_decode += time.perf_counter() - t0
                if kept_windows and kept_windows % 200 == 0:
                    print(
                        f"⏱ decode={t_decode:.1f}s feat+dirtyfeat={t_feat:.1f}s aug={t_aug:.1f}s windows={kept_windows}")
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

                # ✅ Also cap per-file (prevents val/test dominated by a single recording)
                if per_file_limit is not None and kept_windows_in_file >= per_file_limit:
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
                    t0 = time.perf_counter()
                    clean_entry = _mfcc_image_from_chunk(norm_chunk)
                    t_feat += time.perf_counter() - t0

                    # 2. DIRTY (Augment the normalized chunk)
                    t0 = time.perf_counter()
                    dirty_chunk = augment(samples=norm_chunk, sample_rate=SR).astype(np.float32, copy=False)
                    t_aug += time.perf_counter() - t0

                    if len(dirty_chunk) > SAMPLES_PER_CHUNK:
                        dirty_chunk = dirty_chunk[:SAMPLES_PER_CHUNK]
                    elif len(dirty_chunk) < SAMPLES_PER_CHUNK:
                        dirty_chunk = np.pad(
                            dirty_chunk, (0, SAMPLES_PER_CHUNK - len(dirty_chunk))
                        ).astype(np.float32, copy=False)

                    # Keep dirty chunk in a realistic waveform range
                    dirty_chunk = np.clip(dirty_chunk, -1.0, 1.0).astype(np.float32, copy=False)

                    t0 = time.perf_counter()
                    dirty_entry = _mfcc_image_from_chunk(dirty_chunk)
                    t_feat += time.perf_counter() - t0

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
                    kept_windows_in_file += 1
                    if kept_windows % 200 == 0:
                        print(
                            f"⏱ {reciter}: decode={t_decode:.1f}s feat={t_feat:.1f}s aug={t_aug:.1f}s (kept_windows={kept_windows})")

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


    # 🔎 Save group metadata so we can trace failures back to files
    groups_meta_path = Path(OUTPUT_PATH).with_name("groups_meta.json")
    with open(groups_meta_path, "w", encoding="utf-8") as f:
        json.dump(group_meta, f, indent=2)
    print(f"🧾 Wrote groups meta: {groups_meta_path}")

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
