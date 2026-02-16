import os

import numpy as np
from pydub import AudioSegment

# Path to your audio folders
DATA_PATH = "../datasets/audio/"


def get_true_duration(file_path):
    """
    Decodes the audio file to get the exact duration in seconds.
    Slower than reading headers, but 100% accurate for VBR/corrupt MP3s.
    """
    try:
        # Load the file (this actually reads the audio data)
        audio = AudioSegment.from_file(file_path)
        # pydub returns duration in milliseconds
        return len(audio) / 1000.0
    except Exception as e:
        print(f"⚠️ Error decoding {os.path.basename(file_path)}: {e}")
        return 0


def audit():
    print(f"🔍 Scanning {DATA_PATH} using Pydub (Deep Scan)...\n")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Folder '{DATA_PATH}' not found.")
        return

    reciters = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    stats = []

    for reciter in reciters:
        reciter_path = os.path.join(DATA_PATH, reciter)
        files = [f for f in os.listdir(reciter_path) if f.lower().endswith(('.mp3', '.wav'))]

        total_seconds = 0
        file_count = 0

        print(f"  > Decoding {reciter} ({len(files)} files)...", end="\r")

        for f in files:
            dur = get_true_duration(os.path.join(reciter_path, f))
            total_seconds += dur
            file_count += 1

        stats.append({
            "name": reciter,
            "files": file_count,
            "minutes": total_seconds / 60
        })

    # Sort by duration (lowest first)
    stats.sort(key=lambda x: x["minutes"])

    # Calculate statistics
    durations = [s["minutes"] for s in stats]
    if not durations:
        print("\n❌ No audio files found.")
        return

    avg_dur = np.mean(durations)
    median_dur = np.median(durations)

    print("\n\n📊 TRUE DURATION REPORT (DECODED)")
    print("=" * 65)
    print(f"{'RECITER':<30} | {'FILES':<5} | {'MINUTES':<10} | {'STATUS'}")
    print("-" * 65)

    for s in stats:
        flag = "✅ OK"
        # Using Median is safer for skewed datasets
        if s["minutes"] < median_dur * 0.6:
            flag = "⚠️ LOW"
        elif s["minutes"] > median_dur * 2.0:
            flag = "⚠️ HIGH"

        print(f"{s['name']:<30} | {s['files']:<5} | {s['minutes']:<10.2f} | {flag}")

    print("=" * 65)
    print(f"Total Reciters: {len(stats)}")
    print(f"Average Duration: {avg_dur:.2f} min")
    print(f"Median Duration:  {median_dur:.2f} min")
    print("=" * 65)


if __name__ == "__main__":
    audit()
