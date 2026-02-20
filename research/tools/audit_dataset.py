#!/usr/bin/env python3
"""Dataset audit (production gate).

- Deep-decodes audio via pydub to get true durations (handles VBR MP3 correctly).
- Reports per-class file counts + minutes.
- Optional hard-fail thresholds for CI (min_files / min_minutes).

Usage:
  python research/tools/audit_dataset.py --data datasets/audio
  python research/tools/audit_dataset.py --data datasets/audio --min_files 4 --min_minutes 8 --fail
"""

import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from pydub import AudioSegment


AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


def get_true_duration_seconds(file_path: str) -> float:
    try:
        audio = AudioSegment.from_file(file_path)
        return float(len(audio)) / 1000.0
    except Exception as e:
        print(f"⚠️ Error decoding {os.path.basename(file_path)}: {e}")
        return 0.0


@dataclass
class Stat:
    name: str
    files: int
    minutes: float


def audit(data_path: str) -> List[Stat]:
    print(f"🔍 Scanning {data_path} using Pydub (Deep Scan)...\n")

    if not os.path.exists(data_path):
        raise SystemExit(f"❌ Error: Folder '{data_path}' not found.")

    reciters = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    stats: List[Stat] = []

    for reciter in reciters:
        reciter_path = os.path.join(data_path, reciter)
        files = [f for f in os.listdir(reciter_path) if f.lower().endswith(AUDIO_EXTS) and not f.startswith(".")]

        total_seconds = 0.0
        file_count = 0

        print(f"  > Decoding {reciter} ({len(files)} files)...", end="\r")

        for f in files:
            total_seconds += get_true_duration_seconds(os.path.join(reciter_path, f))
            file_count += 1

        stats.append(Stat(name=reciter, files=file_count, minutes=total_seconds / 60.0))

    stats.sort(key=lambda x: x.minutes)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit dataset durations and file counts")
    ap.add_argument("--data", default="datasets/audio", help="Dataset root (class folders)")
    ap.add_argument("--min_files", type=int, default=0, help="Fail if any class has fewer files than this")
    ap.add_argument("--min_minutes", type=float, default=0.0, help="Fail if any class has fewer minutes than this")
    ap.add_argument("--fail", action="store_true", help="Exit non-zero if thresholds violated")
    args = ap.parse_args()

    stats = audit(args.data)

    durations = [s.minutes for s in stats]
    if not durations:
        raise SystemExit("\n❌ No audio files found.")

    avg_dur = float(np.mean(durations))
    median_dur = float(np.median(durations))

    print("\n\n📊 TRUE DURATION REPORT (DECODED)")
    print("=" * 80)
    print(f"{'RECITER':<30} | {'FILES':<5} | {'MINUTES':<10} | {'STATUS'}")
    print("-" * 80)

    violations = []

    for s in stats:
        flag = "✅ OK"
        if s.minutes < median_dur * 0.6:
            flag = "⚠️ LOW"
        elif s.minutes > median_dur * 2.0:
            flag = "⚠️ HIGH"

        if args.min_files and s.files < args.min_files:
            violations.append(f"{s.name}: files {s.files} < {args.min_files}")
        if args.min_minutes and s.minutes < args.min_minutes:
            violations.append(f"{s.name}: minutes {s.minutes:.2f} < {args.min_minutes:.2f}")

        print(f"{s.name:<30} | {s.files:<5d} | {s.minutes:<10.2f} | {flag}")

    print("=" * 80)
    print(f"Total Reciters: {len(stats)}")
    print(f"Average Duration: {avg_dur:.2f} min")
    print(f"Median Duration:  {median_dur:.2f} min")
    print("=" * 80)

    if violations:
        print("\n❌ Threshold violations:")
        for v in violations:
            print(f"   - {v}")
        if args.fail:
            raise SystemExit(2)
        else:
            print("⚠️ (Not failing: pass --fail to enforce in CI)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
