#!/usr/bin/env python3
"""Build a per-class "golden" regression suite quickly.

Goal:
- Ensure the golden suite has >= N files per class so evaluate_model.py doesn't report
  dozens of zero-support classes.

This is NOT meant to be a statistically independent benchmark.
It is a regression guardrail: "same inputs" should not suddenly get worse.

Usage (from repo root):
  python research/tools/build_golden_suite.py \
    --src datasets/audio \
    --dst datasets/audio_test_sets/golden_autofill \
    --n 2 \
    --seed 42

Notes:
- Copies files as-is (mp3/wav). librosa in evaluate_model.py supports both.
- Deterministic selection per class using seed.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def is_audio(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build golden/ test suite by copying N files per class")
    ap.add_argument("--src", default="datasets/audio", help="Source dataset root (class folders)")
    ap.add_argument("--dst", default="datasets/audio_test_sets/golden_autofill", help="Destination golden suite folder")
    ap.add_argument("--n", type=int, default=2, help="Files per class (default: 2)")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    ap.add_argument("--clean", action="store_true", help="Delete dst before copying")
    ap.add_argument("--background_n", type=int, default=None, help="Override N for _background")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    if args.clean and dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")])

    copied_total = 0
    for class_dir in class_dirs:
        name = class_dir.name
        files = sorted([p for p in class_dir.iterdir() if is_audio(p)])
        if not files:
            print(f"⚠️  Skipping {name}: no audio files")
            continue

        # deterministic shuffle per class
        rng = np.random.default_rng(args.seed + (abs(hash(name)) % 100000))
        files = list(files)
        rng.shuffle(files)

        n = args.n
        if name == "_background" and args.background_n is not None:
            n = args.background_n

        picked = files[: min(n, len(files))]
        out_dir = dst / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for p in picked:
            out_path = out_dir / p.name
            # avoid collisions: if same filename exists, add suffix
            if out_path.exists():
                stem = p.stem
                out_path = out_dir / f"{stem}__dup{p.suffix}"
            shutil.copy2(p, out_path)
            copied_total += 1

        print(f"✅ {name}: copied {len(picked)} file(s)")

    print(f"\n📦 Golden suite built at: {dst}")
    print(f"   Total files copied: {copied_total}")
    print("\nNext steps:")
    print("  1) python research/evaluate_model.py --suite golden --update_baseline --min_class_coverage 1.0")
    print("  2) Commit datasets/audio_test_sets/golden_autofill (or store it in a test-data bucket)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
