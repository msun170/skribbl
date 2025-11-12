#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def normalize_name(s: str) -> str:
    # unify spaces/underscores, lowercase, trim
    s = str(s).strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def truthy(v):
    return str(v).strip() not in ("0", "False", "false", "")

def load_allowed_from_matches(matches_csv: Path, min_sim: float, dataset_name: str) -> set:
    """
    Return a set of normalized dataset-class names that matched skribbl words
    with cosine ≥ min_sim for a specific dataset (sketchy or quickdraw).
    """
    import pandas as pd
    df = pd.read_csv(matches_csv)
    # columns: skribbl_raw, skribbl_norm, best_match_raw, best_match_norm, best_match_dataset, cosine_similarity, is_match
    df = df[df["is_match"].astype(bool)]
    df = df[df["best_match_dataset"].str.lower() == dataset_name.lower()]
    df = df[df["cosine_similarity"].astype(float) >= float(min_sim)]

    # prefer the normalized candidate if present, else raw
    if "best_match_norm" in df.columns and df["best_match_norm"].notna().any():
        cands = df["best_match_norm"].fillna(df["best_match_raw"])
    else:
        cands = df["best_match_raw"]

    return {normalize_name(x) for x in cands.tolist()}

def read_invalid_lists(paths):
    bad = set()
    for p in paths:
        if not p:
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                s = s.replace(",", "-")
                bad.add(s)
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_csv", required=True, help="Path to Sketchy stats.csv")
    ap.add_argument("--sketchy_root", required=True, help="Root folder containing class folders (e.g., sketchy_png256/sketches or sketchy/sketches)")
    ap.add_argument("--out_csv", default="sketchy_manifest.csv")
    ap.add_argument("--invalid_lists", nargs="*", default=[], help="optional invalid-*.txt lists")
    ap.add_argument("--drop_flagged", action="store_true", help="drop rows where any quality flag is 1")
    ap.add_argument("--ext", default=".png", help="image extension (.png)")
    # NEW:
    ap.add_argument("--matches_csv", default="skribbl_semantic_matches.csv")
    ap.add_argument("--min_sim", type=float, default=0.60)
    ap.add_argument("--keep_only_matched", action="store_true", default=True,
                    help="Keep only rows whose Category is in the matched set (default: True)")
    args = ap.parse_args()

    sketchy_root = Path(args.sketchy_root)
    bad_ids = read_invalid_lists(args.invalid_lists)

    # Build allowed set from matches (dataset = sketchy)
    allowed = load_allowed_from_matches(Path(args.matches_csv), args.min_sim, dataset_name="sketchy")

    kept = 0
    missing = 0
    skipped_flags = 0
    skipped_bad = 0

    with open(args.stats_csv, "r", encoding="utf-8") as fin, \
         open(args.out_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = [
            "path",
            "title",
            "label",          # keep dataset label (Category) as-is
            "imagenet_id",
            "sketch_id",
            "source",
            "worker_tag",
            "bad_listed",
            "error_flag",
            "context_flag",
            "ambiguous_flag",
            "wrongpose_flag"
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            category = row["Category"].strip()
            category_norm = normalize_name(category)

            # Keep only matched classes (by Category) if requested
            if args.keep_only_matched and category_norm not in allowed:
                continue

            imagenet_id = str(row["ImageNetID"]).strip()
            sketch_id = str(row["SketchID"]).strip()
            title = f"{imagenet_id}-{sketch_id}"

            err = truthy(row.get("Error?", "0"))
            ctx = truthy(row.get("Context?", "0"))
            amb = truthy(row.get("Ambiguous?", "0"))
            pose = truthy(row.get("WrongPose?", "0"))
            is_bad = title.replace(",", "-") in bad_ids

            if args.drop_flagged and (err or ctx or amb or pose):
                skipped_flags += 1
                continue
            if is_bad:
                skipped_bad += 1
                continue

            # Disk existence check
            abs_path = (sketchy_root / category / f"{title}{args.ext}")
            if not abs_path.exists():
                missing += 1
                continue

            rel_path = abs_path
            writer.writerow({
                "path": str(rel_path).replace("\\", "/"),
                "title": title,
                "label": category,           # NO RELABELING
                "imagenet_id": imagenet_id,
                "sketch_id": sketch_id,
                "source": "sketchy",
                "worker_tag": row.get("WorkerTag", "").strip(),
                "bad_listed": int(is_bad),
                "error_flag": int(err),
                "context_flag": int(ctx),
                "ambiguous_flag": int(amb),
                "wrongpose_flag": int(pose),
            })
            kept += 1

    print(f"✓ Wrote {kept} rows to {args.out_csv}")
    if skipped_bad:
        print(f"• Skipped {skipped_bad} listed as invalid.")
    if skipped_flags and args.drop_flagged:
        print(f"• Skipped {skipped_flags} due to quality flags.")
    if missing:
        print(f"• {missing} images were missing on disk (not written).")
    print("Done.")

if __name__ == "__main__":
    main()

'''
python build_sketchy_manifest.py --stats_csv stats.csv --sketchy_root sketchy_png256 --out_csv sketchy_manifest.csv --invalid_lists invalid-ambiguous.txt invalid-context.txt invalid-error.txt invalid-pose.txt --drop_flagged --matches_csv skribbl_semantic_matches.csv --min_sim 0.60 --keep_only_matched

'''