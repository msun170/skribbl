#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def normalize_name(s: str) -> str:
    # unify underscores/spaces, lowercase
    s = str(s).strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def load_allowed_from_matches(matches_csv: Path, min_sim: float, dataset_name: str) -> set:
    import pandas as pd
    df = pd.read_csv(matches_csv)
    df = df[df["is_match"].astype(bool)]
    df = df[df["best_match_dataset"].str.lower() == dataset_name.lower()]
    df = df[df["cosine_similarity"].astype(float) >= float(min_sim)]

    if "best_match_norm" in df.columns and df["best_match_norm"].notna().any():
        cands = df["best_match_norm"].fillna(df["best_match_raw"])
    else:
        cands = df["best_match_raw"]
    return {normalize_name(x) for x in cands.tolist()}

def class_from_dir(dir_name: str) -> str:
    # convert folder names like "alarm_clock" → "alarm clock"
    return normalize_name(dir_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quickdraw_root", required=True,
                    help="Root folder containing class subfolders created from NDJSON→PNG conversion (e.g., quickdraw_png)")
    ap.add_argument("--out_csv", default="quickdraw_manifest.csv")
    # NEW:
    ap.add_argument("--matches_csv", default="skribbl_semantic_matches.csv")
    ap.add_argument("--min_sim", type=float, default=0.60)
    ap.add_argument("--keep_only_matched", action="store_true", default=True)
    ap.add_argument("--exts", nargs="*", default=[".png", ".jpg", ".jpeg"])
    args = ap.parse_args()

    quickdraw_root = Path(args.quickdraw_root)
    allowed = load_allowed_from_matches(Path(args.matches_csv), args.min_sim, dataset_name="quickdraw")

    kept = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["path", "title", "label", "source"])
        writer.writeheader()

        for class_dir in sorted([p for p in quickdraw_root.iterdir() if p.is_dir()]):
            label_norm = class_from_dir(class_dir.name)

            # Keep only matched quickdraw classes
            if args.keep_only_matched and label_norm not in allowed:
                continue

            # Iterate images
            for img in class_dir.rglob("*"):
                if not img.is_file():
                    continue
                if img.suffix.lower() not in [e.lower() for e in args.exts]:
                    continue

                writer.writerow({
                    "path": str(img).replace("\\", "/"),
                    "title": img.stem,
                    "label": label_norm.replace(" ", "_").replace("_", " "),  # keep pretty human label (spaces)
                    "source": "quickdraw",
                })
                kept += 1

    print(f"✓ Wrote {kept} rows to {args.out_csv}")
    print("Done.")

if __name__ == "__main__":
    main()

'''
python build_quickdraw_manifest.py --quickdraw_root quickdraw_png256  --out_csv quickdraw_manifest.csv --matches_csv skribbl_semantic_matches.csv --min_sim 0.60 --keep_only_matched

'''