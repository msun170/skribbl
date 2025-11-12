#!/usr/bin/env python3
"""
Convert a Windows-based training manifest to Linux paths.

- Reads an input CSV (e.g., training_manifest.csv)
- Detects the column that stores image paths (auto or --path-column)
- Normalizes slashes, strips Windows drive prefixes, and rebuilds paths
  under a Linux root (default: /workspace/skribbl)
- Tries to anchor on known dataset folder names (quickdraw_png256, sketchy_png256)
- Optionally writes relative paths instead of absolute
- Optionally checks or drops rows whose files don't exist (use on the pod)

Usage examples:
  # Local (Windows): only rewrite strings, no existence checks
  python manifest_to_linux_path.py --in_csv training_manifest.csv --out_csv linux_training_manifest.csv

  # On the pod: validate existence of files, optionally drop missing
  python manifest_to_linux_path.py --in_csv training_manifest.csv --out_csv linux_training_manifest.csv --check-exists --drop-missing

  # If your path column isn't auto-detected:
  python manifest_to_linux_path.py --in_csv training_manifest.csv --out_csv linux_training_manifest.csv -c path

  # If you prefer relative paths (relative to --root):
  python manifest_to_linux_path.py --in_csv training_manifest.csv --out_csv linux_training_manifest.csv --relative
"""
import argparse
import os
import re
from pathlib import PurePosixPath

import pandas as pd

DEFAULT_DATASET_DIRNAMES = [
    "quickdraw_png256",
    "sketchy_png256",
    "quickdraw",
    "sketchy",
]
CANDIDATE_PATH_COLUMNS = ["path", "image_path", "filepath", "file", "image", "img", "img_path"]


def detect_path_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"[error] path column '{explicit}' not found. Available: {list(df.columns)}")
        return explicit
    for c in CANDIDATE_PATH_COLUMNS:
        if c in df.columns:
            return c
    # fallback: try to guess by dtype / name
    for c in df.columns:
        if df[c].dtype == object and ("path" in c.lower() or "file" in c.lower()):
            return c
    raise SystemExit("[error] Could not detect path column. Use --path-column to specify.")


def normalize_to_linux(p: str, root: str, dataset_basenames: list[str]) -> tuple[str, bool]:
    """
    Convert an arbitrary Windows path string to a Linux path under `root`.

    Strategy:
      1) Convert backslashes -> slashes, strip quotes/whitespace.
      2) Try to anchor after the last occurrence of any dataset basename.
      3) If found, reconstructed = root / (tail after that basename).
      4) Else, strip Windows drive prefix like 'C:/', remove 'file://', collapse slashes.
      5) Return (new_path, anchored_flag)
    """
    if not isinstance(p, str):
        p = str(p)

    s = p.strip().strip('"').strip("'")
    if not s:
        return s, False

    s = s.replace("\\", "/")
    s = re.sub(r"^file:/+","", s, flags=re.IGNORECASE)  # drop file:// prefixes

    # Try to anchor on known dataset basenames
    anchor_idx = -1
    for name in dataset_basenames:
        # Find the LAST occurrence to be robust against nested names
        pattern = f"/{name.lower()}/"
        idx = s.lower().rfind(pattern)
        if idx == -1 and s.lower().endswith("/" + name.lower()):
            idx = s.lower().rfind("/" + name.lower())
        if idx > anchor_idx:
            anchor_idx = idx

    if anchor_idx != -1:
        # Keep everything from the anchor, trim the leading slash for join
        tail = s[anchor_idx + 1 :] if s[anchor_idx] == "/" else s[anchor_idx:]
        tail = re.sub(r"/+", "/", tail)
        new_abs = PurePosixPath(root) / PurePosixPath(tail)
        return new_abs.as_posix(), True

    # No anchor found — remove Windows drive like 'C:/'
    s = re.sub(r"^[A-Za-z]:/", "", s)

    # Strip common Windows user-prefix if present
    s = re.sub(r"^Users/[^/]+/skribbl project/", "", s, flags=re.IGNORECASE)

    # Collapse multiple slashes and rebuild
    s = re.sub(r"/+", "/", s)
    new_abs = PurePosixPath(root) / PurePosixPath(s.lstrip("/"))
    return new_abs.as_posix(), False


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite Windows manifest paths to Linux.")
    ap.add_argument("--in_csv", required=True, help="Input CSV (e.g., training_manifest.csv)")
    ap.add_argument("--out_csv", required=True, help="Output CSV (e.g., linux_training_manifest.csv)")
    ap.add_argument("-c", "--path-column", default=None, help="Name of the path column (auto-detected if omitted)")
    ap.add_argument("--root", default="/workspace/skribbl", help="Linux project root to prepend")
    ap.add_argument(
        "--dataset-dirs",
        nargs="*",
        default=DEFAULT_DATASET_DIRNAMES,
        help=f"Dataset dir basenames to anchor on (default: {DEFAULT_DATASET_DIRNAMES})",
    )
    ap.add_argument("--relative", action="store_true", help="Write paths relative to --root instead of absolute")
    ap.add_argument("--check-exists", dest="check_exists", action="store_true", help="Check that files exist (use on pod)")
    ap.add_argument("--drop-missing", dest="drop_missing", action="store_true", help="Drop rows for missing files (requires --check-exists)")
    args = ap.parse_args()

    # Read CSV (quiet mixed-type warning)
    df = pd.read_csv(args.in_csv, low_memory=False)
    path_col = detect_path_col(df, args.path_column)

    fixed = anchored = missing = 0
    new_paths: list[str] = []

    for raw in df[path_col].astype(str).tolist():
        newp, used_anchor = normalize_to_linux(raw, args.root, args.dataset_dirs)
        if newp != raw:
            fixed += 1
        if used_anchor:
            anchored += 1
        if args.check_exists and not os.path.exists(newp):
            missing += 1
        new_paths.append(newp)

    df[path_col] = new_paths

    if args.check_exists:
        print(f"[check] total rows: {len(df)} | fixed: {fixed} | anchored: {anchored} | missing: {missing}")
        if args.drop_missing and missing:
            before = len(df)
            df = df[df[path_col].map(os.path.exists)]
            print(f"[drop] removed {before - len(df)} missing rows | kept: {len(df)}")

    if args.relative:
        root_pp = PurePosixPath(args.root)
        rels: list[str] = []
        for p in df[path_col].astype(str).tolist():
            try:
                rel = PurePosixPath(p).relative_to(root_pp).as_posix()
            except Exception:
                rel = p  # leave absolute if it's not under root
            rels.append(rel)
        df[path_col] = rels
        print(f"[note] wrote relative paths (base: {args.root})")

    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote: {args.out_csv}")
    print(f"[stats] rows={len(df)} fixed={fixed} anchored={anchored}" + (f" missing={missing}" if args.check_exists else ""))


if __name__ == "__main__":
    main()


'''
python manifest_to_linux_path.py --in_csv training_manifest.csv --out_csv linux_training_manifest.csv
'''