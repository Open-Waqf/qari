import numpy as np
import json
import librosa
from pathlib import Path

# DSP Constants
SR = 22050
FRAME_LENGTH = 512
HOP_LENGTH = 256

def load_matrices():
    with open("models/audio_config.json", "r") as f:
        cfg = json.load(f)
    return {
        "dft_real": np.array(cfg["dft_real"], dtype=np.float32),
        "dft_imag": np.array(cfg["dft_imag"], dtype=np.float32),
        "mel_basis": np.array(cfg["mel_basis"], dtype=np.float32),
        "dct_matrix": np.array(cfg["dct_matrix"], dtype=np.float32),
        "window": np.array(cfg["window"], dtype=np.float32),
    }

MATRICES = load_matrices()

def extract_mfcc(chunk):
    # chunk is 2s @ 22050 = 44100 samples
    frames = librosa.util.frame(chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).T
    out = []
    for frame in frames:
        windowed = frame * MATRICES["window"]
        real = MATRICES["dft_real"] @ windowed
        imag = MATRICES["dft_imag"] @ windowed
        mag = np.sqrt(real**2 + imag**2)
        mel = MATRICES["mel_basis"] @ mag
        log_mel = np.log(mel + 1e-6)
        mfcc = MATRICES["dct_matrix"] @ log_mel
        out.append(mfcc)
    return np.mean(out, axis=0) # Return the "centroid" of this chunk

def analyze_drift():
    # 1. Load Training Centroids
    data = np.load("models/features.npz", allow_pickle=True)
    X = data["X"] # (N, 40, 171, 1)
    y = data["y"]
    is_clean = data["is_clean"]
    mapping = data["mapping"].item()
    inv_map = {v: k for k, v in mapping.items()}

    print("📊 Computing Training Centroids (Clean)...")
    class_centroids = {}
    for name, idx in mapping.items():
        mask = (y == idx) & (is_clean == True)
        if not np.any(mask): continue
        # Mean across samples and time-frames
        # X is (N, 40, 171, 1)
        centroid = np.mean(X[mask], axis=(0, 2, 3))
        class_centroids[idx] = centroid

    # 2. Extract Phone Mic Centroids
    phone_dir = Path("datasets/audio_test_sets/phone_mic")
    phone_files = list(phone_dir.rglob("*.wav"))
    
    # We also need the normalization stats used in features.npz
    # (mean/std from train_clean_only)
    with open("../app/public/models/normalization.json", "r") as f:
        norm = json.load(f)
        n_mean = norm["mean"]
        n_std = norm["std"]

    print(f"\n{'Phone File':<30} | {'Closest Class':<25} | {'Dist'}")
    print("-" * 75)

    for pf in phone_files:
        audio, _ = librosa.load(pf, sr=SR)
        # Take first 2s
        chunk = audio[:44100]
        if len(chunk) < 44100: continue
        
        # Extract and Normalize (must match training pipeline)
        raw_centroid = extract_mfcc(chunk)
        norm_centroid = (raw_centroid - n_mean) / n_std
        
        # Find nearest neighbor
        best_dist = float('inf')
        best_cls = -1
        
        dists = []
        for idx, tr_centroid in class_centroids.items():
            dist = np.linalg.norm(norm_centroid - tr_centroid)
            dists.append((idx, dist))
            if dist < best_dist:
                best_dist = dist
                best_cls = idx
        
        expected = pf.parent.name
        print(f"{pf.name:<30} | {inv_map[best_cls]:<25} | {best_dist:.4f}")
        
        # If misclassified, show where the 'real' class was in the rank
        if inv_map[best_cls] != expected:
            dists.sort(key=lambda x: x[1])
            rank = [inv_map[d[0]] for d in dists].index(expected) + 1
            print(f"   ⚠️ MISSED! {expected} was rank {rank}")
            
            # Feature analysis: which MFCCs are most different?
            expected_idx = mapping[expected]
            diff = norm_centroid - class_centroids[expected_idx]
            top_diff_idx = np.argsort(np.abs(diff))[-3:]
            print(f"   🔥 Top drifting MFCCs vs {expected}: {top_diff_idx} diffs: {diff[top_diff_idx]}")

if __name__ == "__main__":
    analyze_drift()
