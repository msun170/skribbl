#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())

def pick(cands, cols):
    for c in cands:
        if c in cols: return c
    raise KeyError(f"None of {cands} in columns: {list(cols)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_csv", default="skribbl_semantic_matches.csv",
                    help="semantic_coverage output")
    ap.add_argument("--manifest_csv", default="training_manifest.csv",
                    help="merged manifest (path,label,source,title)")
    ap.add_argument("--datasets", nargs="*", default=["quickdraw","sketchy"],
                    help="which datasets count for matches")
    ap.add_argument("--sim", type=float, default=0.60,
                    help="min cosine similarity to consider a match")
    ap.add_argument("--expected_mode", choices=["skribbl","candidates"], default="candidates",
                    help="skribbllabels vs candidate dataset labels as expected")
    ap.add_argument("--out_dir", default="audit_out")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load matches
    m = pd.read_csv(args.matches_csv)
    s_w   = pick(["skribbl_norm","skribbl_raw","word"], m.columns)
    cand  = pick(["best_match_norm","best_match_raw","candidate"], m.columns)
    dset  = pick(["best_match_dataset","dataset","source"], m.columns)
    simc  = pick(["cosine_similarity","similarity","cosine"], m.columns)
    ismc  = pick(["is_match","match"], m.columns)

    mm = m[(m[ismc].astype(bool)) &
           (m[simc].astype(float) >= args.sim) &
           (m[dset].str.lower().isin([d.lower() for d in args.datasets]))].copy()

    # Load manifest
    man = pd.read_csv(args.manifest_csv)
    lab = "label" if "label" in man.columns else pick(["category","class"], man.columns)
    src = "source" if "source" in man.columns else pick(["dataset"], man.columns)
    manifest_labels = {norm(x) for x in man[lab].astype(str)}
    per_label_counts = Counter([norm(x) for x in man[lab].astype(str)])

    # Build “expected” depending on mode
    skribbl_words = sorted({norm(x) for x in mm[s_w].astype(str)})
    if args.expected_mode == "skribbl":
        expected = skribbl_words
        present  = sorted(manifest_labels)
        missing  = sorted(set(expected) - set(present))
        extra    = sorted(set(present) - set(expected))

        print(f"Expected labels (SKRIBBL words): {len(expected)}")
        print(f"Labels present in manifest     : {len(present)}")
        print(f"Missing from manifest          : {len(missing)}")
        if missing: print("  e.g.", missing[:args.show])
        print(f"Present but not expected       : {len(extra)}")
        if extra: print("  e.g.", extra[:args.show])

        # Write basics
        pd.Series(expected, name="class").to_csv(out_dir/"expected_classes.csv", index=False)
        pd.Series(present,  name="class").to_csv(out_dir/"present_classes.csv", index=False)
        pd.Series(missing,  name="class").to_csv(out_dir/"missing_classes.csv", index=False)

    else:  # expected_mode = "candidates"
        # For each skribbl word, collect candidate dataset labels
        buckets = defaultdict(set)
        for _, r in mm.iterrows():
            buckets[norm(r[s_w])].add(norm(r[cand]))

        rows = []
        covered, uncovered = [], []
        for w in skribbl_words:
            cands = sorted(buckets.get(w, []))
            present_cands = [c for c in cands if c in manifest_labels]
            missing_cands = [c for c in cands if c not in manifest_labels]
            is_covered = len(present_cands) > 0
            rows.append({
                "skribb_word": w,
                "num_candidates": len(cands),
                "present_candidates": "|".join(present_cands),
                "missing_candidates": "|".join(missing_cands),
                "covered_by_any_candidate": int(is_covered),
            })
            (covered if is_covered else uncovered).append(w)

        df = pd.DataFrame(rows).sort_values(["covered_by_any_candidate","skribb_word"], ascending=[False, True])
        df.to_csv(out_dir/"skribb_coverage_by_candidates.csv", index=False)

        print(f"Skribbl words with ≥{args.sim:.2f} match in {args.datasets}: {len(skribbl_words)}")
        print(f"Covered by at least one candidate label in manifest : {len(covered)}")
        print(f"NOT covered (no candidate label present)            : {len(uncovered)}")
        if uncovered: print("  e.g.", uncovered[:args.show])

        # Helpful: What are the most frequent manifest labels among candidate sets?
        cand_count = Counter()
        for w in skribbl_words:
            for c in buckets[w]:
                if c in manifest_labels:
                    cand_count[c] += 1
        top_present = pd.DataFrame(cand_count.most_common(50), columns=["label","num_skribbl_words_covered"])
        top_present.to_csv(out_dir/"top_present_candidate_labels.csv", index=False)

    # Save per-class sample counts
    pd.DataFrame(
        sorted(per_label_counts.items()), columns=["label","count"]
    ).to_csv(out_dir/"per_manifest_label_counts.csv", index=False)

if __name__ == "__main__":
    main()

'''
python audit_label_coverage.py --matches_csv skribbl_semantic_matches.csv --manifest_csv training_manifest.csv --datasets quickdraw sketchy --sim 0.60 --expected_mode candidates --out_dir audit_out

'''

# for override aware:

'''
python audit_label_coverage_overrides.py --matches_csv skribbl_semantic_matches.csv --manifest_csv training_manifest.csv --min_sim 0.60 --datasets quickdraw sketchy --overrides_json coverage_overrides.json

'''