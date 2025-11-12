#!/usr/bin/env python3
import argparse
import pandas as pd
import re
from collections import defaultdict

def normalize(s):
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def simple_variants(word):
    """Return a small set of morphological variants for robustness without heavy deps."""
    w = word
    out = set([w])
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        out.add(w[:-1] + "ies")   # party -> parties
    if w.endswith("ies"):
        out.add(w[:-3] + "y")     # parties -> party
    if w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])           # cats -> cat
    else:
        out.add(w + "s")          # cat -> cats
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases_csv", default="aliases.csv",
                    help="Input aliases csv with columns: class,alias,extra_aliases")
    ap.add_argument("--matches_csv", default="skribbl_semantic_matches.csv",
                    help="CSV with columns including: skribbl_raw, best_match_raw, best_match_dataset, cosine_similarity, is_match")
    ap.add_argument("--dataset_whitelist", nargs="*", default=[],
                    help="If set, only accept matches from these datasets (e.g., quickdraw imagenet sketchy)")
    ap.add_argument("--min_sim", type=float, default=0.60)
    ap.add_argument("--max_per_class", type=int, default=5,
                    help="Max number of new aliases to add from matches (beyond existing values)")
    ap.add_argument("--include_variants", action="store_true",
                    help="Also inject simple plural/singular variants for alias & class")
    ap.add_argument("--out_csv", default="aliases_augmented.csv")
    args = ap.parse_args()

    ali = pd.read_csv(args.aliases_csv)
    # Column picks
    def pick(colnames, columns):
        look = [c for c in colnames if c in columns]
        if not look:
            raise KeyError(f"None of {colnames} found in columns: {list(columns)}")
        return look[0]

    class_col = pick(["class", "label", "canonical"], ali.columns)
    alias_col = pick(["alias", "primary_alias"], ali.columns)
    extra_col = "extra_aliases" if "extra_aliases" in ali.columns else None
    if extra_col is None:
        ali["extra_aliases"] = ""
        extra_col = "extra_aliases"

    # Build map from class -> current aliases set
    current_aliases = {}
    for _, row in ali.iterrows():
        base = normalize(row[alias_col]) if pd.notna(row[alias_col]) else ""
        extras = [normalize(x) for x in str(row.get(extra_col, "") or "" ).split("|") if x.strip()]
        s = set([normalize(row[class_col])])  # ensure class itself kept
        if base: s.add(base)
        s.update(extras)
        current_aliases[normalize(row[class_col])] = s

    # Load matches
    m = pd.read_csv(args.matches_csv)
    # Column picks for matches
    skribbl_col = pick(["skribbl_raw", "skribbl_norm", "word"], m.columns)
    match_col   = pick(["best_match_raw", "best_match_norm", "candidate"], m.columns)
    dataset_col = pick(["best_match_dataset", "dataset", "source"], m.columns)
    sim_col     = pick(["cosine_similarity", "similarity", "cosine"], m.columns)
    ismatch_col = pick(["is_match", "match"], m.columns)

    # Aggregate candidates per skribbl word
    buckets = defaultdict(list)
    for _, r in m.iterrows():
        if not bool(r[ismatch_col]):
            continue
        if float(r[sim_col]) < args.min_sim:
            continue
        if args.dataset_whitelist:
            if normalize(r[dataset_col]) not in [normalize(d) for d in args.dataset_whitelist]:
                continue
        w = normalize(r[skribbl_col])
        cand = normalize(r[match_col])
        sim = float(r[sim_col])
        buckets[w].append((cand, sim))

    # Build augmented aliases
    out_rows = []
    for _, row in ali.iterrows():
        klass = normalize(row[class_col])
        base_alias = normalize(row[alias_col])
        extras = [normalize(x) for x in str(row.get(extra_col, "")).split("|") if x and x.strip()]

        bag = set([klass])
        if base_alias: bag.add(base_alias)
        bag.update(extras)

        if args.include_variants:
            bag |= set().union(*[simple_variants(x) for x in list(bag)])

        # Add top-N candidates from matches
        cands = sorted(buckets.get(klass, []), key=lambda x: -x[1])
        added = 0
        for cand, sim in cands:
            if cand not in bag:
                bag.add(cand)
                added += 1
            if added >= args.max_per_class:
                break

        # Recompose fields
        alias_out = row[alias_col]
        class_out = row[class_col]
        bag_norm = sorted(bag)
        bag_norm = [b for b in bag_norm if b not in {normalize(class_out), normalize(alias_out)}]
        extra_out = "|".join(bag_norm)

        newrow = row.copy()
        newrow[alias_col] = alias_out
        newrow["extra_aliases"] = extra_out
        out_rows.append(newrow)

    out = pd.DataFrame(out_rows, columns=ali.columns)
    out.to_csv(args.out_csv, index=False)
    print(f"✓ Wrote augmented aliases -> {args.out_csv}")

if __name__ == "__main__":
    main()
