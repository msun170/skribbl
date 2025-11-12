#!/usr/bin/env python3
import argparse, json, re
import pandas as pd
from pathlib import Path

def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"[()'/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def all_variants(s: str):
    base = norm(s)
    return {base, base.replace(" ", "_"), base.replace(" ", "-")}

def pick(cands, cols):
    for c in cands:
        if c in cols: return c
    raise KeyError(f"Missing any of {cands} in {list(cols)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_csv", default="skribbl_semantic_matches.csv")
    ap.add_argument("--manifest_csv", default="training_manifest.csv")
    ap.add_argument("--min_sim", type=float, default=0.60)
    ap.add_argument("--datasets", nargs="*", default=["quickdraw", "sketchy"])
    ap.add_argument("--overrides_json", default="coverage_overrides.json",
                    help="Manual mapping from Skribbl words → dataset labels.")
    args = ap.parse_args()

    # Load overrides
    overrides = {}
    path = Path(args.overrides_json)
    if path.exists():
        overrides = {norm(k): [norm(v) for v in vs] 
                     for k, vs in json.loads(path.read_text(encoding="utf-8")).items()}

    m = pd.read_csv(args.matches_csv)
    man = pd.read_csv(args.manifest_csv)

    skribbl_col = pick(["skribbl_norm","skribbl_raw","word"], m.columns)
    ds_col      = pick(["best_match_dataset","dataset","source"], m.columns)
    cand_col    = pick(["best_match_raw","best_match_norm","candidate"], m.columns)
    sim_col     = pick(["cosine_similarity","similarity","cosine"], m.columns)
    label_col   = pick(["label","category","class"], man.columns)

    labels_present = {norm(x) for x in man[label_col].astype(str)}
    want_ds = {d.lower() for d in args.datasets}

    groups = {}
    for _, r in m.iterrows():
        if float(r[sim_col]) < args.min_sim: 
            continue
        if str(r[ds_col]).lower() not in want_ds:
            continue
        w = norm(r[skribbl_col])
        cand = norm(r[cand_col])
        groups.setdefault(w, set()).add(cand)

    for w, extra in overrides.items():
        groups.setdefault(w, set()).update(extra)

    expanded = {w: set().union(*[all_variants(c) for c in cands])
                for w, cands in groups.items()}

    covered = []
    missing = []
    for w, cands in expanded.items():
        if labels_present.intersection(cands):
            covered.append(w)
        else:
            missing.append(w)

    print(f"Skribbl words with ≥{args.min_sim:.2f} match in {list(want_ds)}: {len(expanded)}")
    print(f"Covered by at least one candidate label in manifest : {len(covered)}")
    print(f"NOT covered (no candidate label present)            : {len(missing)}")
    if missing:
        from itertools import islice
        print("  e.g.", list(islice(missing, 10)))

if __name__ == "__main__":
    main()

"""
python audit_label_coverage_overrides.py --matches_csv skribbl_semantic_matches.csv --manifest_csv training_manifest.csv --overrides_json coverage_overrides.json --datasets quickdraw sketchy --min_sim 0.6

"""