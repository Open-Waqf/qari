#!/usr/bin/env python3
"""
Extract background audio clips for:
- training:   datasets/audio/_background
- golden:     datasets/audio_test_sets/golden/_background
- challenge:  datasets/audio_test_sets/challenge/_background

Defaults:
- SR=22050 (to match your training pipeline)
- Always resets target folder
- Ensures test backgrounds are DIFFERENT from training by excluding ESC-50 source files used by training.
- Writes a manifest JSON in each output folder with the ESC-50 filenames used.

Usage:
  # training (default)
  python tools/extract_background.py

  # golden background (different from training)
  python tools/extract_background.py --preset golden

  # challenge background (different from training + golden)
  python tools/extract_background.py --preset challenge
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

# -----------------------------
# ✅ GLOBAL DEFAULTS
# -----------------------------
SR = 22050  # match your prepare_data SR
PCM_SUBTYPE = "PCM_16"
RES_TYPE = "soxr_hq"

# Mild safeguard so "silence" isn't *too* silent (still basically room tone)
MIN_RMS_BG = 0.002
MAX_GAIN_BG = 20.0

# ESC-50 location (adjust if needed)
DEFAULT_ESC50_PATH = "../datasets/sound_datasets/esc50/ESC-50-master"

# Output roots (relative to script run location)
DEFAULT_TRAIN_BG_DIR = "../datasets/audio/_background"
DEFAULT_TEST_ROOT = "../datasets/audio_test_sets"

# Preset configs (minimal args; you can change these constants)
PRESET_CONFIG = {
    "train": {
        "out_dir": DEFAULT_TRAIN_BG_DIR,
        "noise_mins": 60.0,
        "silence_mins": 10.0,
        "clip_sec": 10.0,
        "seed": 42,
        "categories": "train_full",  # use full relevant set below
    },
    "golden": {
        "out_dir": f"{DEFAULT_TEST_ROOT}/golden/_background",
        "noise_mins": 5.0,
        "silence_mins": 2.0,
        "clip_sec": 20.0,
        "seed": 1337,
        "categories": "golden_subset",
    },
    "challenge": {
        "out_dir": f"{DEFAULT_TEST_ROOT}/challenge/_background",
        "noise_mins": 15.0,
        "silence_mins": 2.0,
        "clip_sec": 20.0,
        "seed": 2026,
        "categories": "train_full",
    },
}

# -----------------------------
# 🎯 RELEVANT ESC-50 CATEGORIES
# (same idea as your original script)
# -----------------------------
RELEVANT_CATEGORIES_FULL = {
    # Indoor / Domestic
    "keyboard_typing", "mouse_click", "door_wood_knock", "door_wood_creaks",
    "clock_tick", "vacuum_cleaner", "washing_machine", "can_opening",

    # Human / Body
    "coughing", "sneezing", "breathing", "footsteps", "laughing",
    "clapping", "snoring", "crying_baby", "drinking_sipping",

    # Urban / Exterior
    "car_horn", "engine", "train", "siren", "wind", "rain", "thunderstorm",
    "crickets", "chirping_birds",
}

# Golden: keep it “realistic & common” but not too chaotic
RELEVANT_CATEGORIES_GOLDEN = {
    "keyboard_typing", "mouse_click", "clock_tick",
    "breathing", "coughing", "sneezing",
    "door_wood_knock", "door_wood_creaks", "footsteps",
    "rain", "wind",
}


# -----------------------------
# 🧹 Helper: Safe reset folder
# -----------------------------
def reset_folder(folder: Path):
    """Deletes all files in the folder to start fresh. Guarded."""
    folder = folder.resolve()

    # Safety guard: only allow deleting if path includes "_background"
    if "_background" not in folder.as_posix():
        raise RuntimeError(f"Refusing to reset non-_background folder: {folder}")

    if not folder.exists():
        print(f"ℹ️ Folder does not exist yet: {folder}")
        return

    print(f"🧹 Resetting folder: {folder}")
    deleted = 0
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
            deleted += 1
    print(f"✨ Folder clean. Deleted {deleted} files.")


# -----------------------------
# 📖 ESC-50 Metadata
# -----------------------------
def load_esc50_metadata(esc50_root: Path) -> Dict[str, str]:
    """
    Looks for meta/esc50.csv and returns filename->category mapping.
    """
    csv_path = esc50_root / "meta" / "esc50.csv"
    if not csv_path.exists():
        csv_path = esc50_root.parent / "meta" / "esc50.csv"

    if not csv_path.exists():
        print(f"⚠️ ESC-50 metadata not found at {csv_path}. Categories will be 'noise'.")
        return {}

    mapping: Dict[str, str] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["filename"]] = row["category"]
        print(f"📖 Loaded ESC-50 metadata: {len(mapping)} rows.")
    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")
    return mapping


# -----------------------------
# 🔊 DSP helpers
# -----------------------------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _normalize_min_rms(x: np.ndarray, min_rms: float = MIN_RMS_BG, max_gain: float = MAX_GAIN_BG) -> np.ndarray:
    r = _rms(x)
    if r < 1e-8:
        return x
    if r >= min_rms:
        return x
    g = min(max_gain, min_rms / r)
    return np.clip(x * g, -1.0, 1.0).astype(np.float32, copy=False)


def _make_clip(y: np.ndarray, sr: int, clip_sec: float, rng: random.Random) -> np.ndarray:
    n = int(round(clip_sec * sr))
    if y.size == 0:
        return np.zeros((n,), dtype=np.float32)

    # Loop if too short
    if y.size < n:
        repeats = int(np.ceil(n / y.size))
        y = np.tile(y, repeats)

    # Random slice
    if y.size > n:
        start = rng.randint(0, y.size - n)
        out = y[start:start + n]
    else:
        out = y

    # Random gain (0.6x to 1.4x)
    g = 0.6 + rng.random() * 0.8
    out = np.clip(out * g, -1.0, 1.0).astype(np.float32, copy=False)

    # Ensure it's not *too* silent (still realistic)
    out = _normalize_min_rms(out)
    return out


def generate_silence(out_dir: Path, minutes: float, sr: int, rng: np.random.Generator):
    total_samples = int(minutes * 60 * sr)

    # Very low noise (room tone)
    y = (rng.standard_normal(total_samples).astype(np.float32) * 0.0008)
    y = _normalize_min_rms(y)

    chunk_len = 10 * sr
    num_chunks = total_samples // chunk_len

    print(f"🤫 Generating {minutes:.1f} mins of room tone @ {sr}Hz ({num_chunks} files)")
    for i in range(num_chunks):
        chunk = y[i * chunk_len: (i + 1) * chunk_len]
        path = out_dir / f"silence_synthetic_{i:03d}.wav"
        sf.write(path, chunk, sr, subtype=PCM_SUBTYPE)


# -----------------------------
# 🧾 Manifest helpers (to enforce disjoint sets)
# -----------------------------
def manifest_path(out_dir: Path) -> Path:
    return out_dir / "_manifest.json"


def load_used_sources(manifest_file: Path) -> Set[str]:
    if not manifest_file.exists():
        return set()
    try:
        with manifest_file.open("r", encoding="utf-8") as f:
            j = json.load(f)
        return set(j.get("used_esc50_files", []))
    except Exception:
        return set()


def write_manifest(out_dir: Path, meta: dict, used_files: List[str]):
    payload = {
        "meta": meta,
        "used_esc50_files": used_files,
        "created_at_unix": int(time.time()),
    }
    with manifest_path(out_dir).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"🧾 Wrote manifest: {manifest_path(out_dir).resolve()}")


def get_exclusion_set(preset: str) -> Set[str]:
    """
    For test presets, exclude any ESC-50 sources used by train (and golden if challenge).
    This ensures test background is different from training background.
    """
    exclude: Set[str] = set()

    train_dir = Path(PRESET_CONFIG["train"]["out_dir"]).resolve()
    exclude |= load_used_sources(manifest_path(train_dir))

    if preset == "challenge":
        golden_dir = Path(PRESET_CONFIG["golden"]["out_dir"]).resolve()
        exclude |= load_used_sources(manifest_path(golden_dir))

    return exclude


# -----------------------------
# 🎧 ESC-50 processing
# -----------------------------
def find_esc50_audio_dir(esc_root: Path) -> Path:
    # typical ESC-50 layout: ESC-50-master/audio/*.wav
    audio_dir = esc_root / "audio"
    if audio_dir.exists():
        return audio_dir
    return esc_root


def select_category_set(mode: str) -> Set[str]:
    if mode == "golden_subset":
        return RELEVANT_CATEGORIES_GOLDEN
    return RELEVANT_CATEGORIES_FULL


def process_esc50(
        esc50_root: Path,
        out_dir: Path,
        target_mins: float,
        clip_sec: float,
        sr: int,
        rng_py: random.Random,
        exclude_sources: Set[str],
        categories: Set[str],
) -> Tuple[List[str], int]:
    meta_map = load_esc50_metadata(esc50_root)
    audio_dir = find_esc50_audio_dir(esc50_root)

    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        print(f"❌ No ESC-50 wav files found in {audio_dir.resolve()}")
        return [], 0

    # shuffle deterministically (seeded)
    rng_py.shuffle(files)

    required_samples = int(target_mins * 60 * sr)
    current_samples = 0
    written = 0
    used_sources: List[str] = []
    skipped_irrelevant = 0
    skipped_excluded = 0

    print(f"📂 ESC-50 scan: {len(files)} files | target={target_mins:.1f} mins | clip={clip_sec:.1f}s | sr={sr}")
    print(f"🚫 Excluding {len(exclude_sources)} ESC-50 sources (to keep sets disjoint)")

    for fpath in files:
        if current_samples >= required_samples:
            break

        src_name = fpath.name  # e.g., 1-100032-A-0.wav
        if src_name in exclude_sources:
            skipped_excluded += 1
            continue

        category = meta_map.get(src_name, "noise")

        # Only keep relevant categories if known
        if category != "noise" and category not in categories:
            skipped_irrelevant += 1
            continue

        try:
            y, _ = librosa.load(str(fpath), sr=sr, mono=True, res_type=RES_TYPE)
            y = y.astype(np.float32, copy=False)

            clip = _make_clip(y, sr, clip_sec, rng_py)

            out_name = f"{category}_{fpath.stem}.wav"
            sf.write(out_dir / out_name, clip, sr, subtype=PCM_SUBTYPE)

            current_samples += clip.size
            written += 1
            used_sources.append(src_name)

        except Exception as e:
            print(f"⚠️ Error reading {fpath.name}: {e}")

    mins_done = current_samples / sr / 60.0
    print(f"✅ ESC-50 done: {written} clips | {mins_done:.2f} mins produced")
    print(f"🗑️ Skipped: {skipped_irrelevant} irrelevant, {skipped_excluded} excluded (overlap-avoidance)")
    return used_sources, written


# -----------------------------
# 🚀 Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["train", "golden", "challenge"], default="train")
    p.add_argument("--esc50_path", default=DEFAULT_ESC50_PATH)
    args = p.parse_args()

    cfg = PRESET_CONFIG[args.preset]
    out_dir = Path(cfg["out_dir"]).resolve()
    esc50_root = Path(args.esc50_path).resolve()

    # Deterministic RNGs
    rng_py = random.Random(cfg["seed"])
    rng_np = np.random.default_rng(cfg["seed"])

    print("\n" + "=" * 70)
    print(f"🎛️  PRESET: {args.preset.upper()}")
    print(f"📁 Output:  {out_dir}")
    print(f"🎚️  SR:      {SR}")
    print(f"⏱️  Noise:   {cfg['noise_mins']} mins | Silence: {cfg['silence_mins']} mins | Clip: {cfg['clip_sec']} sec")
    print(f"🎲 Seed:    {cfg['seed']}")
    print("=" * 70)

    # Always reset folder (as you requested)
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_folder(out_dir)

    # Build exclusion set for test presets
    exclude_sources: Set[str] = set()
    if args.preset in ("golden", "challenge"):
        exclude_sources = get_exclusion_set(args.preset)
        if not exclude_sources:
            print("⚠️ No training manifest found yet, so overlap-avoidance is limited.")
            print("   Tip: run training preset once first: python tools/extract_background.py")

    # Choose categories
    cat_set = select_category_set(cfg["categories"])
    print(f"🏷️  Categories enabled: {len(cat_set)}")

    # 1) silence
    if cfg["silence_mins"] > 0:
        generate_silence(out_dir, cfg["silence_mins"], SR, rng_np)

    # 2) ESC-50 noise
    used_files: List[str] = []
    if cfg["noise_mins"] > 0:
        if not esc50_root.exists():
            print(f"⚠️ ESC-50 not found at: {esc50_root}")
            print("   Only silence was generated.")
        else:
            used_files, _ = process_esc50(
                esc50_root=esc50_root,
                out_dir=out_dir,
                target_mins=float(cfg["noise_mins"]),
                clip_sec=float(cfg["clip_sec"]),
                sr=SR,
                rng_py=rng_py,
                exclude_sources=exclude_sources,
                categories=cat_set,
            )

    # Write manifest so future presets can avoid overlap
    meta = {
        "preset": args.preset,
        "sr": SR,
        "seed": cfg["seed"],
        "clip_sec": cfg["clip_sec"],
        "noise_mins": cfg["noise_mins"],
        "silence_mins": cfg["silence_mins"],
        "esc50_root": str(esc50_root),
        "overlap_avoidance": True if args.preset in ("golden", "challenge") else False,
        "excluded_sources_count": len(exclude_sources),
    }
    write_manifest(out_dir, meta, used_files)

    print("\n✅ DONE.")
    print(f"📦 Background folder ready: {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
