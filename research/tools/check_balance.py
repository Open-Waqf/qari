#!/usr/bin/env python3
import argparse
from collections import Counter

import numpy as np


def balanced_weight(total_samples: int, num_classes: int, class_count: int) -> float:
    # Same as sklearn's "balanced": n_samples / (n_classes * n_samples_in_class)
    return total_samples / (num_classes * class_count)


def main():
    ap = argparse.ArgumentParser(description="Inspect class balance + weights inside features.npz")
    ap.add_argument("--file", default="models/features.npz", help="Path to features.npz")
    ap.add_argument("--cap", type=int, default=None, help="If provided, flag classes with count == cap")
    ap.add_argument("--show-keys", action="store_true", help="Print NPZ keys")
    args = ap.parse_args()

    file = args.file
    d = np.load(file, allow_pickle=True)

    if "y" not in d or "mapping" not in d:
        raise SystemExit("features.npz must contain at least keys: 'y' and 'mapping'")

    y = d["y"]
    X = d["X"]
    print("X.shape =", X.shape)  # should be (N, 40, 186, 1)
    print("width =", X.shape[2])
    mapping = d["mapping"].item()  # name -> idx
    inv = {int(v): str(k) for k, v in mapping.items()}

    n = int(len(y))
    classes = sorted(inv.keys())
    k = int(len(classes))

    # Optional: groups (filenames / ids) if present
    groups = d["groups"] if "groups" in d.files else None

    print(f"Check balance from {file}")
    if args.show_keys:
        print("NPZ keys:", list(d.files))

    # Counts
    c = Counter(y.tolist())

    # Summary stats
    counts = [c.get(cls, 0) for cls in classes]
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    mean_count = (sum(counts) / k) if k else 0.0
    ratio = (max_count / min_count) if min_count else float("inf")

    print(f"classes: {k}")
    print(f"total:   {n}")
    print(f"min/mean/max: {min_count} / {mean_count:.1f} / {max_count}")
    print(f"imbalance ratio (max/min): {ratio:.2f}x" if ratio != float("inf") else "imbalance ratio: inf")

    # Table (sorted by count ascending like your current script)
    rows = []
    for cls, cnt in sorted(c.items(), key=lambda kv: kv[1]):
        name = inv.get(int(cls), f"idx_{int(cls)}")
        pct = (100.0 * cnt / n) if n else 0.0
        w = balanced_weight(n, k, cnt) if cnt else 0.0

        row = {
            "idx": int(cls),
            "name": name,
            "count": int(cnt),
            "pct": pct,
            "w_bal": w,
        }

        # If groups exist, compute unique group count per class
        if groups is not None and len(groups) == len(y):
            mask = (y == cls)
            uniq = len(set(groups[mask].tolist()))
            row["uniq_groups"] = uniq

        rows.append(row)

    # Header
    header = ["class", "count", "%", "w_balanced"]
    if any("uniq_groups" in r for r in rows):
        header.append("uniq_groups")
    if args.cap is not None:
        header.append("hit_cap")

    print("\n" + "  ".join(f"{h:>12s}" for h in header))
    print("  ".join("-" * 12 for _ in header))

    for r in rows:
        hit_cap = (args.cap is not None and r["count"] == args.cap)
        parts = [
            f"{r['name']:<12s}" if len(r["name"]) <= 12 else f"{r['name'][:12]:<12s}",
            f"{r['count']:12d}",
            f"{r['pct']:11.2f}%",
            f"{r['w_bal']:12.3f}",
        ]
        if "uniq_groups" in r:
            parts.append(f"{r['uniq_groups']:12d}")
        if args.cap is not None:
            parts.append(f"{('YES' if hit_cap else ''):>12s}")

        print("  ".join(parts))

    # Extra: spotlight background if present
    if "_background" in mapping:
        bidx = int(mapping["_background"])
        bcnt = c.get(bidx, 0)
        bpct = (100.0 * bcnt / n) if n else 0.0
        bw = balanced_weight(n, k, bcnt) if bcnt else 0.0
        print("\n_background spotlight:")
        print(f"  idx={bidx}, count={bcnt}, pct={bpct:.2f}%, w_balanced={bw:.3f}")

    # Cap hint (auto if cap not supplied)
    if args.cap is None and max_count > 0:
        # common pattern: many classes equal to same max value => likely cap
        freq = Counter(counts)
        most_common_count, how_many = freq.most_common(1)[0]
        if how_many >= 3 and most_common_count == max_count:
            capped = [inv[cls] for cls in classes if c.get(cls, 0) == max_count]
            print("\nPossible cap detected:")
            print(f"  {how_many} classes share the max count of {max_count}.")
            print("  capped-like classes:", ", ".join(capped))


if __name__ == "__main__":
    main()
