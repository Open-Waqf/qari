import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# 🎚️ DEFAULTS
# -----------------------------
SR = 22050
WINDOW_SEC_DEFAULT = 2.0
HOP_SEC_DEFAULT = 1.0  # 50% overlap for 2s window (optional but consistent)
RMS_MIN_DEFAULT = 0.01  # match prepare_data.py filter for alignment
RMS_MIN_RECITER = 0.01
RMS_MIN_BG = 0.003


# -----------------------------
# 🧠 MODEL UTILS (Custom Layer)
# -----------------------------
@keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    """Matches train.py custom layer signature. At inference, it's a no-op."""

    def __init__(self, freq_mask_param=3, time_mask_param=12, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_mask_param": self.freq_mask_param, "time_mask_param": self.time_mask_param})
        return cfg


# -----------------------------
# 🧮 DSP PARITY: MATRICES
# -----------------------------
@dataclass
class Matrices:
    dft_real: np.ndarray
    dft_imag: np.ndarray
    mel_basis: np.ndarray
    dct_matrix: np.ndarray
    window: np.ndarray


def load_matrices(audio_config_path: Path) -> Matrices:
    with audio_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    def arr(name: str) -> np.ndarray:
        return np.array(cfg[name], dtype=np.float32)

    return Matrices(
        dft_real=arr("dft_real"),
        dft_imag=arr("dft_imag"),
        mel_basis=arr("mel_basis"),
        dct_matrix=arr("dct_matrix"),
        window=arr("window"),
    )


FRAME_LENGTH = 512
HOP_LENGTH = 256


def extract_mfcc_image(chunk: np.ndarray, m: Matrices) -> np.ndarray:
    """Return MFCC image shape (40,T) from a chunk using frame=512 hop=256."""
    frames = librosa.util.frame(chunk, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).T  # (T,512)
    out = np.empty((frames.shape[0], 40), dtype=np.float32)

    for i, frame in enumerate(frames):
        windowed = frame * m.window
        real = m.dft_real @ windowed
        imag = m.dft_imag @ windowed
        mag = np.sqrt(real ** 2 + imag ** 2)
        mel = m.mel_basis @ mag
        log_mel = np.log(mel + 1e-6)
        mfcc = m.dct_matrix @ log_mel
        out[i, :] = mfcc

    img = out.T  # (40,T)
    expected_frames = 1 + (len(chunk) - FRAME_LENGTH) // HOP_LENGTH
    if img.shape != (40, expected_frames):
        raise ValueError(f"Bad MFCC shape {img.shape}, expected (40,{expected_frames})")
    return img


# -----------------------------
# 🎛️ DSP PARITY: SIGNAL CHAIN
# -----------------------------
@dataclass
class DspConfig:
    target_rms: float = 0.10
    min_gain: float = 0.6
    max_gain: float = 2.5
    rms_floor: float = 0.002
    pre_emph: float = 0.95


def calculate_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def normalize_signal(x: np.ndarray, cfg: DspConfig) -> np.ndarray:
    r = calculate_rms(x)

    # 1. Floor Gate
    if r < cfg.rms_floor:
        return x

    # 2. Linear Gain
    g = cfg.target_rms / max(1e-12, r)
    g = float(np.clip(g, cfg.min_gain, cfg.max_gain))

    if abs(g - 1.0) < 1e-3:
        return x

    # 🟢 PARITY FIX: Linear Gain + Hard Clip
    y = x * g
    y = np.clip(y, -1.0, 1.0)  # Hard clip prevents float32 blowouts
    return y.astype(np.float32, copy=False)


def pre_emphasis(x: np.ndarray, a: float) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y


# -----------------------------
# 📊 METRICS & REPORTING
# -----------------------------
@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    macro_f1_seen: float
    macro_f1_all: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]


@dataclass
class FileResult:
    path: str
    expected: str
    predicted: str
    confidence: float
    windows_used: int


def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Metrics:
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1

    acc = float(np.trace(cm) / max(1, np.sum(cm)))

    per = {}
    f1s_all = []
    f1s_seen = []

    for i, name in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        support = float(np.sum(cm[i, :]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per[name] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

        f1s_all.append(f1)
        if support > 0:
            f1s_seen.append(f1)

    macro_f1_all = float(np.mean(f1s_all)) if f1s_all else 0.0
    macro_f1_seen = float(np.mean(f1s_seen)) if f1s_seen else 0.0

    return Metrics(
        accuracy=acc,
        macro_f1=macro_f1_seen,
        macro_f1_seen=macro_f1_seen,
        macro_f1_all=macro_f1_all,
        per_class=per,
        confusion=cm.tolist(),
    )


def print_report(metrics: Metrics, results: List[FileResult]):
    print("\n🔍 DETAILED REPORT")
    print("-" * 60)

    print(f"{'CLASS':<25} | {'RECALL':<8} | {'SUPPORT':<5}")
    print("-" * 45)
    sorted_classes = sorted(metrics.per_class.items(), key=lambda x: x[1]["recall"])
    for cls, stats in sorted_classes:
        if stats["support"] > 0:
            print(f"{cls:<25} | {stats['recall']:.1%}   | {int(stats['support']):<5}")

    print("\n❌ MISCLASSIFIED FILES")
    print("-" * 60)
    errors = [r for r in results if r.predicted != r.expected]

    if not errors:
        print("🎉 None! Perfect score.")
        return

    errors.sort(key=lambda x: x.confidence, reverse=True)
    for e in errors:
        fname = Path(e.path).name
        print(f"📄 {fname}")
        print(f"   Expected:  {e.expected}")
        print(f"   Predicted: {e.predicted} ({e.confidence:.1%})")
        print(f"   Windows:   {e.windows_used}")
        print("-" * 30)


# -----------------------------
# 🚀 EVALUATION
# -----------------------------
def load_labels_map(reciters_map_path: Path) -> List[str]:
    with reciters_map_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return [str(x) for x in labels]


def load_norm_stats(norm_path: Path) -> Tuple[float, float]:
    if not norm_path.exists():
        print("⚠️ Normalization file not found, defaulting to (0,1)")
        return 0.0, 1.0
    with norm_path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    return float(j.get("mean", 0.0)), max(float(j.get("std", 1.0)), 1e-6)


def iter_audio_files(test_dir: Path) -> List[Tuple[str, Path]]:
    out = []
    if not test_dir.exists():
        return out

    for label_dir in sorted([p for p in test_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        label = label_dir.name
        for f in sorted(label_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac"):
                if not f.name.startswith("."):
                    out.append((label, f))
    return out


def evaluate_suite(
        suite_name: str,
        suite_dir: Path,
        model_path: Path,
        audio_config_path: Path,
        reciters_map_path: Path,
        normalization_path: Path,
        window_sec: float,
        hop_sec: float,
        rms_min: float,
        apply_rms_norm: bool,
        apply_pre_emph: bool,
        max_windows_per_file: int,
) -> Tuple[Metrics, List[FileResult]]:
    matrices = load_matrices(audio_config_path)
    labels = load_labels_map(reciters_map_path)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    mean, std = load_norm_stats(normalization_path)

    print(f"📦 Loading model: {model_path.name}")
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={"SpecAugment": SpecAugment},
        compile=False,
    )

    dsp = DspConfig()
    pairs = iter_audio_files(suite_dir)

    if not pairs:
        print(f"❌ [{suite_name}] No audio files found in {suite_dir}.")
        return Metrics(0, 0, 0, 0, {}, []), []

    y_true: List[int] = []
    y_pred: List[int] = []
    results: List[FileResult] = []

    print(f"\n==================== SUITE: {suite_name.upper()} ====================")
    print(f"🔍 Evaluating {len(pairs)} files in {suite_dir} ...")

    win = int(round(window_sec * SR))
    hop = int(round(hop_sec * SR))

    print(f"🔧 Eval config: SR={SR}, window={window_sec}s ({win} samples), hop={hop_sec}s ({hop} samples)")

    for expected_label, path in pairs:
        if expected_label not in label_to_idx:
            print(f"⚠️ Skipping {path}: folder '{expected_label}' not in reciters_map.json")
            continue

        try:
            audio, _ = librosa.load(str(path), sr=SR, mono=True, res_type="soxr_hq")
            audio = audio.astype(np.float32, copy=False)
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")
            continue

        if audio.shape[0] < win:
            audio = np.pad(audio, (0, win - audio.shape[0])).astype(np.float32, copy=False)

        windows = []
        used = 0
        for s in range(0, audio.shape[0] - win + 1, hop):
            chunk = audio[s: s + win]
            if chunk.shape[0] != win:
                continue
            rms_gate = RMS_MIN_BG if expected_label == "_background" else RMS_MIN_RECITER

            if calculate_rms(chunk) < rms_gate:
                continue

            if apply_rms_norm:
                chunk = normalize_signal(chunk, dsp)
            if apply_pre_emph:
                chunk = pre_emphasis(chunk, dsp.pre_emph)

            img = extract_mfcc_image(chunk, matrices)
            img = (img - mean) / std

            windows.append(img)
            used += 1
            if max_windows_per_file > 0 and used >= max_windows_per_file:
                break

        if not windows:
            print(f"⚠️ {path.name}: 0 windows kept (rms_gate={rms_gate}).")
            continue

        X = np.stack(windows, axis=0).astype(np.float32)[..., np.newaxis]  # (N,40,186,1)
        probs = model.predict(X, verbose=0)  # (N, num_classes)

        avg_probs = np.mean(probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])

        y_true.append(label_to_idx[expected_label])
        y_pred.append(pred_idx)

        results.append(FileResult(
            path=str(path),
            expected=expected_label,
            predicted=labels[pred_idx],
            confidence=confidence,
            windows_used=len(windows),
        ))

    # ===== Background-specific metrics =====
    if "_background" in label_to_idx:
        bg_idx = label_to_idx["_background"]
        yt = np.array(y_true, dtype=np.int64)
        yp = np.array(y_pred, dtype=np.int64)

        far = float(np.mean((yt == bg_idx) & (yp != bg_idx)))  # bg -> reciter
        frr = float(np.mean((yt != bg_idx) & (yp == bg_idx)))  # reciter -> bg

        rec_mask = (yt != bg_idx)
        rec_only_acc = float(np.mean(yp[rec_mask] == yt[rec_mask])) if np.any(rec_mask) else 0.0

        bg_mask = (yt == bg_idx)
        bg_recall = float(np.mean(yp[bg_mask] == bg_idx)) if np.any(bg_mask) else 0.0

        print("\n===== Background Metrics =====")
        print(f"FAR (bg -> reciter): {far:.4f}")
        print(f"FRR (reciter -> bg): {frr:.4f}")
        print(f"Reciter-only acc:    {rec_only_acc:.4f}")
        print(f"BG recall:           {bg_recall:.4f}")
    else:
        print("\n(no _background in reciters_map.json; skipping FAR/FRR)")

    metrics = compute_metrics(y_true, y_pred, labels)
    return metrics, results


# -----------------------------
# 💾 BASELINES
# -----------------------------
def save_baseline(path: Path, metrics: Metrics, meta: dict):
    data = {
        "meta": meta,
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "per_class": metrics.per_class,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compare_to_baseline(metrics: Metrics, baseline_path: Path, strict: bool) -> Tuple[bool, List[str]]:
    if not baseline_path.exists():
        return True, ["No baseline found."]

    with baseline_path.open("r", encoding="utf-8") as f:
        base = json.load(f).get("metrics", {})

    b_acc = float(base.get("accuracy", 0.0))
    b_f1 = float(base.get("macro_f1", 0.0))

    msgs = []
    ok = True
    eps = 1e-4

    if metrics.accuracy < b_acc - eps:
        ok = False
        msgs.append(f"Accuracy regressed: {metrics.accuracy:.4f} < {b_acc:.4f}")

    if metrics.macro_f1 < b_f1 - eps:
        ok = False
        msgs.append(f"Macro-F1 regressed: {metrics.macro_f1:.4f} < {b_f1:.4f}")

    if strict:
        b_pc = base.get("per_class", {})
        for cls, stats in metrics.per_class.items():
            if cls in b_pc:
                cur_rec = float(stats["recall"])
                base_rec = float(b_pc[cls].get("recall", 0.0))
                if cur_rec < base_rec - eps:
                    ok = False
                    msgs.append(f"Recall regressed for {cls}: {cur_rec:.2f} < {base_rec:.2f}")

    return ok, msgs


def fail_on_high_conf_errors(results: List[FileResult], threshold: float) -> Tuple[bool, List[str]]:
    if threshold is None:
        return True, []
    bad = [r for r in results if r.predicted != r.expected and r.confidence >= threshold]
    if not bad:
        return True, []
    msgs = [
        f"High-confidence wrong prediction: {Path(r.path).name} expected={r.expected} predicted={r.predicted} conf={r.confidence:.2%}"
        for r in bad]
    return False, msgs


# -----------------------------
# 🏁 CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Golden + Challenge Evaluator (No-Regression Gate)")

    p.add_argument("--root_dir", default="datasets/audio_test_sets",
                   help="Root folder containing golden/ and challenge/")
    p.add_argument("--suite", choices=["golden", "challenge", "both"], default="both")

    p.add_argument("--model", default="models/qari_model.keras", help="Path to the model to evaluate")
    p.add_argument("--audio_config", default="../app/public/models/audio_config.json")
    p.add_argument("--reciters_map", default="../app/public/models/reciters_map.json")
    p.add_argument("--normalization", default="../app/public/models/normalization.json")

    p.add_argument("--baseline_golden", default="models/golden_baseline.json")
    p.add_argument("--baseline_challenge", default="models/challenge_baseline.json")

    p.add_argument("--window_sec", type=float, default=WINDOW_SEC_DEFAULT)
    p.add_argument("--hop_sec", type=float, default=HOP_SEC_DEFAULT)
    p.add_argument("--max_windows", type=int, default=40)
    p.add_argument("--rms_min", type=float, default=RMS_MIN_DEFAULT)

    p.add_argument("--update_baseline", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no_rms_norm", action="store_true")
    p.add_argument("--pre_emph", action="store_true", help="Enable pre-emphasis (match app toggle)")
    p.add_argument("--fail_on_high_conf_error", type=float, default=None,
                   help="If set, fail when any wrong prediction has confidence >= this value (e.g. 0.70)")

    args = p.parse_args()
    cwd = Path.cwd()

    root = (cwd / args.root_dir).resolve()
    suites = ["golden", "challenge"] if args.suite == "both" else [args.suite]

    suite_to_baseline = {
        "golden": (cwd / args.baseline_golden).resolve(),
        "challenge": (cwd / args.baseline_challenge).resolve(),
    }

    overall_ok = True

    for suite in suites:
        suite_dir = root / suite
        baseline_path = suite_to_baseline[suite]

        metrics, results = evaluate_suite(
            suite_name=suite,
            suite_dir=suite_dir,
            model_path=(cwd / args.model).resolve(),
            audio_config_path=(cwd / args.audio_config).resolve(),
            reciters_map_path=(cwd / args.reciters_map).resolve(),
            normalization_path=(cwd / args.normalization).resolve(),
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            rms_min=args.rms_min,
            apply_rms_norm=not args.no_rms_norm,
            apply_pre_emph=args.pre_emph,
            max_windows_per_file=args.max_windows,
        )

        if not results:
            print(f"\n❌ [{suite}] FAILED: No files were actually evaluated. Check folder names and contents.")
            overall_ok = False
            continue

        print_report(metrics, results)

        print("\n" + "=" * 60)
        print(
            f"📊 RESULTS: Acc={metrics.accuracy:.2%} | F1(seen)={metrics.macro_f1_seen:.2%} | F1(all)={metrics.macro_f1_all:.2%}")
        print("=" * 60)

        # Optional hard fail on high-confidence wrong preds
        ok_hc, msgs_hc = fail_on_high_conf_errors(results, args.fail_on_high_conf_error)
        if not ok_hc:
            overall_ok = False
            print(f"❌ [{suite}] FAILED: High-confidence misclassification(s):")
            for m in msgs_hc:
                print(f"   - {m}")

        if args.update_baseline:
            meta = vars(args).copy()
            meta["suite"] = suite
            save_baseline(baseline_path, metrics, meta)
            print(f"✅ [{suite}] Baseline UPDATED at {baseline_path.name}")
        else:
            ok, msgs = compare_to_baseline(metrics, baseline_path, args.strict)
            if ok:
                print(f"✅ [{suite}] PASSED: No regression detected vs {baseline_path.name}.")
            else:
                overall_ok = False
                print(f"❌ [{suite}] FAILED: Regression detected vs {baseline_path.name}!")
                for m in msgs:
                    print(f"   - {m}")

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
