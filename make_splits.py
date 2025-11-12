#!/usr/bin/env python3
import argparse, hashlib, json, math, random
from pathlib import Path
import pandas as pd

def pick(colnames, columns):
    for c in colnames:
        if c in columns: return c
    raise KeyError(f"Need one of {colnames}, but have {list(columns)}")

def stable_shuffle(items, seed):
    rnd = random.Random(str(seed))
    items = list(items)
    rnd.shuffle(items)
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--out_splits", default="splits.csv")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--test_ratio",  type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_per_class", type=int, default=1,
                    help="Classes with < this many samples go entirely to train.")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Optional cap per class before splitting (0 = no cap).")
    ap.add_argument("--by_source", action="store_true",
                    help="If set, stratify within (label,source) buckets (stricter).")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    path_col  = pick(["path","filepath","image"], df.columns)
    label_col = pick(["label","category","class"], df.columns)
    src_col   = "source" if "source" in df.columns else None

    groups_key = [label_col, src_col] if (args.by_source and src_col) else [label_col]

    rows = []
    for key, g in df.groupby(groups_key):
        label = key[0] if isinstance(key, tuple) else key
        items = g[path_col].tolist()

        # optional cap per class
        if args.max_per_class and len(items) > args.max_per_class:
            items = stable_shuffle(items, (args.seed, label)).__iter__()
            items = list(list(items)[:args.max_per_class])

        # stable per-class shuffle
        items = stable_shuffle(items, (args.seed, label))

        if len(items) < args.min_per_class:
            # tiny classes -> all train
            for p in items:
                rows.append((p, label, "train"))
            continue

        n = len(items)
        n_train = int(round(n * args.train_ratio))
        n_val   = int(round(n * args.val_ratio))
        n_test  = n - n_train - n_val
        # adjust if rounding hurt
        while n_train + n_val + n_test < n: n_train += 1
        while n_train + n_val + n_test > n and n_test>0: n_test -= 1

        split_items = (
            (items[:n_train], "train"),
            (items[n_train:n_train+n_val], "val"),
            (items[n_train+n_val:], "test"),
        )
        for chunk, s in split_items:
            for p in chunk:
                rows.append((p, label, s))

    out = pd.DataFrame(rows, columns=["path","label","split"])
    out.to_csv(args.out_splits, index=False)
    print(f"✓ Wrote {len(out)} rows -> {args.out_splits}")
    # quick stats
    for s in ["train","val","test"]:
        print(f"{s:>5}: {len(out[out.split==s])}")
    print("Done.")

if __name__ == "__main__":
    main()

'''
python make_splits.py --manifest training_manifest.csv --out_splits splits.csv --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 7
'''