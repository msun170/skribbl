import argparse
import pandas as pd
from pathlib import Path

def pick(colnames, columns):
    look = [c for c in colnames if c in columns]
    if not look:
        raise KeyError(f"None of {colnames} found in columns: {list(columns)}")
    return look[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases_csv", default="aliases.csv")
    ap.add_argument("--out_csv", default="aliases_needs_review.csv")
    ap.add_argument("--only_changed", action="store_true",
                    help="If set, only include rows where alias != class")
    args = ap.parse_args()

    df = pd.read_csv(args.aliases_csv)

    needs_col = pick(["needs_review", "review", "flag_review"], df.columns)
    class_col = pick(["class", "label", "canonical"], df.columns)
    alias_col = pick(["alias", "primary_alias"], df.columns)

    mask = df[needs_col].astype(int) == 1
    if args.only_changed:
        mask = mask & (df[alias_col].astype(str).str.strip().str.lower() !=
                       df[class_col].astype(str).str.strip().str.lower())

    out = df.loc[mask].copy()
    out.to_csv(args.out_csv, index=False)
    print(f"✓ Wrote {len(out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()

'''
python augment_aliases_with_matches.py --aliases_csv aliases.csv --matches_csv skribbl_semantic_matches.csv --min_sim 0.60 --max_per_class 5 --include_variants --out_csv aliases_augmented.csv

'''
