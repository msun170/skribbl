#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path

def norm(s: str) -> str:
    return (s or "").strip()

def load_words(words_path: Path):
    words = []
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words

def load_matches(matches_csv: Path, min_sim: float):
    """
    Returns: dict[str, list[dict]] keyed by skribbl_norm
    Each dict has keys: best_match_norm, best_match_dataset, cosine_similarity, best_match_raw, skribbl_raw
    """
    buckets = defaultdict(list)
    with open(matches_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expected headers:
        # skribbl_raw,skribbl_norm,best_match_raw,best_match_norm,best_match_dataset,cosine_similarity,is_match
        for row in reader:
            try:
                sim = float(row.get("cosine_similarity", "0") or 0.0)
            except ValueError:
                sim = 0.0
            is_match = str(row.get("is_match", "")).strip().lower() in ("1","true","t","yes","y")
            if not is_match or sim < min_sim:
                continue
            s_norm = norm(row.get("skribbl_norm"))
            buckets[s_norm].append({
                "best_match_norm": norm(row.get("best_match_norm")),
                "best_match_dataset": norm(row.get("best_match_dataset")),
                "best_match_raw": norm(row.get("best_match_raw")),
                "cosine_similarity": sim,
                "skribbl_raw": norm(row.get("skribbl_raw")),
            })
    # sort each bucket by descending similarity
    for k in buckets:
        buckets[k].sort(key=lambda d: d["cosine_similarity"], reverse=True)
    return buckets

def choose_alias(cands, class_name: str, confident_sim: float, tie_delta: float):
    """
    Decide alias and whether review is needed.
    - alias defaults to class_name
    - if top candidate has sim >= confident_sim, we *suggest* mapping to that alias
    - needs_review if:
        * no candidates, or
        * there are ≥2 distinct candidate aliases within tie_delta of the top,
        * alias != class_name (so you can eyeball changes),
        * or top alias looks suspiciously different in surface form.
    """
    needs_review = False
    alias = class_name  # default

    if not cands:
        return alias, needs_review or True, [], None, 0

    top = cands[0]
    top_sim = top["cosine_similarity"]
    top_alias = top["best_match_norm"]

    # How many distinct aliases within tie band?
    close_aliases = []
    for c in cands:
        if top_sim - c["cosine_similarity"] <= tie_delta:
            close_aliases.append(c["best_match_norm"])
    n_close_unique = len(set(a for a in close_aliases if a))

    # suggest changing to top alias only if confidently similar and not identical
    if top_sim >= confident_sim and top_alias and (top_alias.lower() != class_name.lower()):
        alias = top_alias
        needs_review = True  # whenever we change alias, we mark for quick human scan

    # If multiple distinct top aliases within tie band, ask for review
    if n_close_unique > 1:
        needs_review = True

    # Build a compact list of extra aliases (unique in score order, excluding alias itself)
    extras = []
    seen = set([alias.lower()])
    for c in cands:
        a = c["best_match_norm"]
        if not a:
            continue
        al = a.lower()
        if al in seen:
            continue
        seen.add(al)
        extras.append(a)

    # Pick the most common dataset among close top candidates (for info)
    ds_counter = Counter([c["best_match_dataset"] for c in cands if (top_sim - c["cosine_similarity"] <= tie_delta)])
    top_ds = (ds_counter.most_common(1)[0][0] if ds_counter else (cands[0]["best_match_dataset"] or ""))

    return alias, needs_review, extras, top_ds, len(set([c["best_match_norm"] for c in cands if c["best_match_norm"]]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True, help="Path to words.txt (label space)")
    ap.add_argument("--matches", required=True, help="Path to skribbl_semantic_matches.csv")
    ap.add_argument("--out", default="aliases.csv", help="Output CSV")
    ap.add_argument("--min_sim", type=float, default=0.60, help="Minimum similarity to consider a candidate alias")
    ap.add_argument("--confident_sim", type=float, default=0.90, help="If top candidate >= this, suggest alias change")
    ap.add_argument("--tie_delta", type=float, default=0.02, help="Within this of the top score counts as 'tied'")
    ap.add_argument("--max_extras", type=int, default=6, help="Cap number of extra_aliases saved")
    args = ap.parse_args()

    words = load_words(Path(args.words))
    buckets = load_matches(Path(args.matches), args.min_sim)

    out_fields = [
        "class",            # canonical class (from words.txt)
        "alias",            # primary alias used in prompts (may equal class)
        "extra_aliases",    # pipe-delimited additional aliases (optional)
        "top_similarity",   # best similarity observed (0 if none)
        "candidate_count",  # number of distinct alias candidates considered
        "top_source",       # dataset most represented near top score band
        "needs_review",     # 1/0 for quick filtering
        "notes",            # freeform (left blank)
    ]

    kept = 0
    flagged = 0
    missing = 0

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()

        for cls in words:
            cls_norm = cls.strip()
            cands = buckets.get(cls_norm, [])

            alias, needs_review, extras, top_ds, cand_count = choose_alias(
                cands, cls_norm, args.confident_sim, args.tie_delta
            )

            top_similarity = cands[0]["cosine_similarity"] if cands else 0.0

            if not cands:
                missing += 1
            if needs_review:
                flagged += 1

            writer.writerow({
                "class": cls_norm,
                "alias": alias,
                "extra_aliases": "|".join(extras[:args.max_extras]),
                "top_similarity": f"{top_similarity:.4f}",
                "candidate_count": cand_count,
                "top_source": top_ds or "",
                "needs_review": int(needs_review),
                "notes": "",
            })
            kept += 1

    print(f"✓ Wrote {kept} rows → {args.out}")
    print(f"• {flagged} rows flagged as needs_review (inspect alias changes / ties).")
    print(f"• {missing} classes had no candidates with similarity ≥ {args.min_sim:.2f}.")

if __name__ == "__main__":
    main()

'''
python suggest_aliases.py --words skribbl_words.txt --matches skribbl_semantic_matches.csv --out aliases.csv --min_sim 0.60 --confident_sim 0.90 --tie_delta 0.02 --max_extras 6

'''