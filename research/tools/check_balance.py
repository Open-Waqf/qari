#!/usr/bin/env python3
"""
tools/check_balance.py

Inspect class balance + weights inside models/features.npz (POST-cap, POST-gating),
and add hard fail checks for clean/dirty pairing + optional meta/config parity checks.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def balanced_weight(total_samples: int, num_classes: int, class_count: int) -> float:
    # Same as sklearn's "balanced": n_samples / (n_classes * n_samples_in_class)
    return total_samples / (num_classes * class_count) if class_count else 0.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_meta_path(npz_path: str) -> str:
    # models/features.npz -> models/features.meta.json
    p = Path(npz_path)
    return str(p.with_suffix(".meta.json"))


def load_meta(meta_path: str) -> Optional[Dict[str, Any]]:
    if not meta_path:
        return None
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def invert_mapping(mapping: Dict[str, int]) -> Tuple[list[str], Dict[int, str]]:
    num_classes = len(mapping)
    names = [None] * num_classes
    for name, idx in mapping.items():
        names[int(idx)] = str(name)
    inv = {i: n for i, n in enumerate(names) if n is not None}
    return names, inv


def hard_pairing_checks(
        y: np.ndarray,
        groups: Optional[np.ndarray],
        is_clean: Optional[np.ndarray],
        fail_hard: bool = True,
) -> None:
    n = len(y)
    if n % 2 != 0:
        msg = f"Expected even N (clean/dirty pairs). Got N={n}."
        raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)

    # If is_clean exists, ensure it is alternating [True, False, True, False...]
    if is_clean is not None:
        if len(is_clean) != n:
            msg = f"is_clean length mismatch: {len(is_clean)} vs y length {n}"
            raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)
        expected = (np.arange(n) % 2 == 0)
        if not np.array_equal(is_clean, expected):
            # Not necessarily fatal if you *intentionally* changed ordering,
            # but then your downstream code must not assume parity.
            msg = "is_clean does not match expected alternating [clean,dirty,...] pattern."
            raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)

    # Labels must match within each pair
    if not np.all(y[0::2] == y[1::2]):
        bad = np.where(y[0::2] != y[1::2])[0][:20]
        msg = f"Broken pairing: label mismatch within pairs at pair indices: {bad.tolist()}"
        raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)

    # Groups must match within each pair (same source file/group id)
    if groups is not None:
        if len(groups) != n:
            msg = f"groups length mismatch: {len(groups)} vs y length {n}"
            raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)
        if not np.all(groups[0::2] == groups[1::2]):
            bad = np.where(groups[0::2] != groups[1::2])[0][:20]
            msg = f"Broken pairing: group mismatch within pairs at pair indices: {bad.tolist()}"
            raise SystemExit(msg) if fail_hard else print("⚠️ " + msg)


def main():
    ap = argparse.ArgumentParser(description="Inspect class balance + weights inside features.npz (post-cap).")
    ap.add_argument("--file", default="models/features.npz", help="Path to features.npz")
    ap.add_argument("--cap", type=int, default=None, help="If provided, flag classes with count == cap")
    ap.add_argument("--show-keys", action="store_true", help="Print NPZ keys")
    ap.add_argument(
        "--fail-hard",
        action="store_true",
        help="Hard-fail on pairing/config issues (recommended for CI).",
    )
    ap.add_argument(
        "--meta",
        default=None,
        help="Optional path to features.meta.json (default: alongside features.npz).",
    )
    ap.add_argument(
        "--audio-config",
        default=None,
        help="Optional path to audio_config.json to hash and compare with meta (parity check).",
    )
    args = ap.parse_args()

    npz_path = args.file
    d = np.load(npz_path, allow_pickle=True)

    required = {"X", "y", "mapping"}
    if not required.issubset(set(d.files)):
        raise SystemExit(f"{npz_path} must contain keys: {sorted(list(required))}. Found: {list(d.files)}")

    X = d["X"]
    y = d["y"].astype(np.int64, copy=False)
    mapping = d["mapping"].item()

    if not isinstance(mapping, dict) or not mapping:
        raise SystemExit("mapping must be a non-empty dict (name -> idx).")

    # Mapping integrity: indices should be 0..C-1
    num_classes = len(mapping)
    if set(int(v) for v in mapping.values()) != set(range(num_classes)):
        raise SystemExit("mapping indices must be contiguous 0..C-1 (stable class ordering required).")

    names, inv = invert_mapping(mapping)
    classes = sorted(inv.keys())

    # Optional: groups and is_clean
    groups = d["groups"].astype(np.int64, copy=False) if "groups" in d.files else None
    is_clean = d["is_clean"].astype(bool, copy=False) if "is_clean" in d.files else None

    print(f"Check balance from {npz_path}")
    if args.show_keys:
        print("NPZ keys:", list(d.files))

    # Shape checks
    print("X.shape =", X.shape)
    if X.ndim != 4:
        msg = f"Expected X to be 4D (N,H,W,C). Got ndim={X.ndim}."
        if args.fail_hard:
            raise SystemExit(msg)
        print("⚠️ " + msg)
    else:
        n, h, w, c = X.shape
        print(f"height={h}, width={w}, channels={c}")
        if h != 40:
            msg = f"Unexpected MFCC height: {h} (expected 40)."
            if args.fail_hard:
                raise SystemExit(msg)
            print("⚠️ " + msg)

    # Pairing sanity (high value for your normalization logic)
    hard_pairing_checks(y=y, groups=groups, is_clean=is_clean, fail_hard=args.fail_hard)
    if is_clean is None:
        is_clean = (np.arange(len(y)) % 2 == 0)

    # Optional meta + config parity
    meta_path = args.meta if args.meta is not None else default_meta_path(npz_path)
    meta = load_meta(meta_path)
    if meta is not None:
        print(f"\n📎 Meta: {meta_path}")
        # Print a few key fields if present
        for k in ["sr", "duration", "frame_length", "hop_length", "max_gain", "min_gain", "target_rms",
                  "audio_config_sha256"]:
            if k in meta:
                print(f"  - {k}: {meta[k]}")
    else:
        print(f"\nℹ️ Meta not found (optional): {meta_path}")

    if args.audio_config:
        if not os.path.exists(args.audio_config):
            msg = f"audio_config.json not found at: {args.audio_config}"
            raise SystemExit(msg) if args.fail_hard else print("⚠️ " + msg)
        else:
            h = sha256_file(args.audio_config)
            print(f"\n🔐 audio_config.json SHA256: {h}")
            if meta and "audio_config_sha256" in meta and meta["audio_config_sha256"] != h:
                msg = "Parity mismatch: audio_config.json hash != meta audio_config_sha256 (features built with different matrices)."
                raise SystemExit(msg) if args.fail_hard else print("⚠️ " + msg)

    n = int(len(y))
    k = int(len(classes))

    # Counts
    ccounts = Counter(y.tolist())
    counts = [ccounts.get(cls, 0) for cls in classes]

    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    mean_count = (sum(counts) / k) if k else 0.0
    ratio = (max_count / min_count) if min_count else float("inf")

    print("\nSummary")
    print(f"classes: {k}")
    print(f"total:   {n}")
    print(f"clean:   {int(is_clean.sum())}   dirty: {int((~is_clean).sum())}")
    print(f"min/mean/max: {min_count} / {mean_count:.1f} / {max_count}")
    print(f"imbalance ratio (max/min): {ratio:.2f}x" if ratio != float("inf") else "imbalance ratio: inf")

    # Table
    rows = []
    for cls in classes:
        cnt = int(ccounts.get(cls, 0))
        name = inv.get(int(cls), f"idx_{int(cls)}")
        pct = (100.0 * cnt / n) if n else 0.0
        w_bal = balanced_weight(n, k, cnt) if cnt else 0.0

        mask = (y == cls)
        cnt_clean = int(np.sum(mask & is_clean))
        cnt_dirty = int(np.sum(mask & (~is_clean)))

        row = {
            "idx": int(cls),
            "name": name,
            "count": cnt,
            "clean": cnt_clean,
            "dirty": cnt_dirty,
            "pct": pct,
            "w_bal": w_bal,
        }

        if groups is not None:
            uniq = len(set(groups[mask].tolist()))
            row["uniq_groups"] = int(uniq)

        rows.append(row)

    # Sort ascending by count
    rows.sort(key=lambda r: r["count"])

    header = ["class", "count", "clean", "dirty", "%", "w_balanced"]
    if any("uniq_groups" in r for r in rows):
        header.append("uniq_groups")
    if args.cap is not None:
        header.append("hit_cap")

    print("\n" + "  ".join(f"{h:>12s}" for h in header))
    print("  ".join("-" * 12 for _ in header))

    for r in rows:
        hit_cap = (args.cap is not None and r["count"] == args.cap)
        name = r["name"]
        parts = [
            f"{name:<12s}" if len(name) <= 12 else f"{name[:12]:<12s}",
            f"{r['count']:12d}",
            f"{r['clean']:12d}",
            f"{r['dirty']:12d}",
            f"{r['pct']:11.2f}%",
            f"{r['w_bal']:12.3f}",
        ]
        if "uniq_groups" in r:
            parts.append(f"{r['uniq_groups']:12d}")
        if args.cap is not None:
            parts.append(f"{('YES' if hit_cap else ''):>12s}")
        print("  ".join(parts))

    # Background spotlight
    if "_background" in mapping:
        bidx = int(mapping["_background"])
        bcnt = int(ccounts.get(bidx, 0))
        bpct = (100.0 * bcnt / n) if n else 0.0
        bw = balanced_weight(n, k, bcnt) if bcnt else 0.0
        print("\n_background spotlight:")
        print(f"  idx={bidx}, count={bcnt}, pct={bpct:.2f}%, w_balanced={bw:.3f}")

    # Cap hint (auto if cap not supplied)
    if args.cap is None and max_count > 0:
        freq = Counter(counts)
        most_common_count, how_many = freq.most_common(1)[0]
        if how_many >= 3 and most_common_count == max_count:
            capped = [inv[cls] for cls in classes if ccounts.get(cls, 0) == max_count]
            print("\nPossible cap detected:")
            print(f"  {how_many} classes share the max count of {max_count}.")
            print("  capped-like classes:", ", ".join(capped))

    # Risk warnings (useful with GroupShuffleSplit)
    if groups is not None:
        low = [r for r in rows if r.get("uniq_groups", 999999) < 4]
        if low:
            print("\n⚠️ Split-risk warning (few unique groups/files):")
            for r in low:
                print(
                    f"  - {r['name']}: uniq_groups={r['uniq_groups']} (val/test may miss this class with group splits)")


if __name__ == "__main__":
    main()
