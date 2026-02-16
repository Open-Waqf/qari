import argparse
import csv
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------
# 🧹 Helper: Clean Folder
# ---------------------------------------------------------
def reset_folder(folder: Path):
    """Deletes all files in the folder to start fresh."""
    if not folder.exists():
        return
    print(f"🧹 Resetting folder: {folder} ...")
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
    print("✨ Folder clean.")


# ---------------------------------------------------------
# 📖 Helper: Read ESC-50 Metadata
# ---------------------------------------------------------
def load_esc50_metadata(esc50_root: Path):
    """
    Looks for meta/esc50.csv and returns a dict: {'1-100.wav': 'dog', ...}
    """
    csv_path = esc50_root / "meta" / "esc50.csv"

    # Try one level up if not found (common unzip issue)
    if not csv_path.exists():
        csv_path = esc50_root.parent / "meta" / "esc50.csv"

    if not csv_path.exists():
        print(f"⚠️ Warning: Metadata not found at {csv_path}. Filenames will be generic.")
        return {}

    mapping = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map filename -> category (e.g. 'dog')
                mapping[row['filename']] = row['category']
        print(f"📖 Loaded metadata for {len(mapping)} files.")
    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")

    return mapping


# ---------------------------------------------------------
# 🔊 DSP Processing
# ---------------------------------------------------------
def _make_clip(y: np.ndarray, sr: int, clip_sec: float) -> np.ndarray:
    n = int(round(clip_sec * sr))

    # 1. LOOP (Tile) if too short
    if y.size < n:
        repeats = int(np.ceil(n / y.size))
        y = np.tile(y, repeats)

    # 2. Random Slice
    if y.size > n:
        start = random.randint(0, y.size - n)
        out = y[start:start + n]
    else:
        out = y

    # 3. Random Gain (0.6x to 1.4x)
    g = 0.6 + random.random() * 0.8
    out = np.clip(out * g, -1.0, 1.0)
    return out.astype(np.float32)


def generate_silence(out_dir: Path, minutes: float, sr: int = 16000):
    total_samples = int(minutes * 60 * sr)
    # Very low noise (room tone)
    y = np.random.randn(total_samples).astype(np.float32) * 0.0008

    chunk_len = 10 * sr
    num_chunks = total_samples // chunk_len

    print(f"🤫 Generating {minutes} mins of room tone ({num_chunks} files)...")
    for i in range(num_chunks):
        chunk = y[i * chunk_len: (i + 1) * chunk_len]
        path = out_dir / f"silence_synthetic_{i:03d}.wav"
        sf.write(path, chunk, sr)


def process_esc50(src_dir: Path, out_dir: Path, target_mins: float, clip_sec: float):
    # 1. Load categories
    meta_map = load_esc50_metadata(src_dir)

    # 2. Find WAVs
    # Try looking in 'audio' subdir first (standard ESC-50 structure)
    audio_dir = src_dir / "audio"
    if not audio_dir.exists():
        audio_dir = src_dir  # Fallback to root

    files = list(audio_dir.glob("*.wav"))
    if not files:
        print(f"❌ No .wav files found in {audio_dir}")
        return

    random.shuffle(files)

    required_samples = int(target_mins * 60 * 16000)
    current_samples = 0
    count = 0

    print(f"📂 Scanning {len(files)} files in {audio_dir.name}...")

    for fpath in files:
        if current_samples >= required_samples:
            break

        try:
            # Load
            y, sr = librosa.load(str(fpath), sr=16000, mono=True)
            y_clip = _make_clip(y, 16000, clip_sec)

            # Name Determination
            # Use CSV category if available, else generic name
            original_name = fpath.name
            category = meta_map.get(original_name, "noise")

            # Clean filename (rain_1-5423.wav)
            out_name = f"{category}_{fpath.stem}.wav"

            # Save
            sf.write(out_dir / out_name, y_clip, 16000)

            current_samples += y_clip.size
            count += 1
        except Exception as e:
            print(f"⚠️ Error reading {fpath.name}: {e}")

    print(f"✅ Extracted {count} clips ({current_samples / 16000 / 60:.2f} mins).")


# ---------------------------------------------------------
# 🚀 Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esc50_path", default="../datasets/sound_datasets/esc50/ESC-50-master")
    p.add_argument("--out_dir", default="../datasets/audio/_background")
    p.add_argument("--noise_mins", type=float, default=30.0)
    p.add_argument("--silence_mins", type=float, default=5.0)
    p.add_argument("--clip_sec", type=float, default=10.0)

    # NEW FLAG
    p.add_argument("--reset", action="store_true", help="Delete existing files in output folder first")

    args = p.parse_args()

    out_path = Path(args.out_dir)

    # 1. Handle Reset
    if args.reset:
        reset_folder(out_path)

    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Generate Silence
    if args.silence_mins > 0:
        generate_silence(out_path, args.silence_mins)

    # 3. Process ESC-50
    esc_path = Path(args.esc50_path)
    if esc_path.exists():
        process_esc50(esc_path, out_path, args.noise_mins, args.clip_sec)
    else:
        print(f"⚠️ Skipping ESC-50: Path not found {esc_path}")


if __name__ == "__main__":
    main()
