import csv, json, argparse, random

DEFAULT_TEMPLATES = [
    "a sketch of a {term}",
    "a simple line drawing of a {term}",
    "a black-and-white doodle of a {term}",
    "a contour drawing of a {term}",
    "a rough pencil sketch of a {term}",
    "a minimalist drawing of a {term}",
    "a quick hand-drawn sketch of a {term}",
    "a childlike drawing of a {term}",
    "a schematic drawing of a {term}",
    "a cartoon-style sketch of a {term}",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aliases_csv", required=True)
    ap.add_argument("--out_json", default="prompts.json")
    ap.add_argument("--templates", nargs="*", default=None,
                    help="Optional custom templates; use {term} placeholder.")
    ap.add_argument("--per_class", type=int, default=12,
                    help="How many prompts to materialize per class (unique strings).")
    args = ap.parse_args()

    templates = args.templates or DEFAULT_TEMPLATES

    out = {}
    with open(args.aliases_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cls = row["class"].strip()
            alias = row["alias"].strip()
            extras = [s.strip() for s in (row.get("extra_aliases", "") or "").split("|") if s.strip()]
            pool = [alias] + extras

            # Build a small set of prompts by mixing templates × terms, then de-dup & cap.
            prompts = []
            for t in templates:
                for term in pool:
                    prompts.append(t.format(term=term))
            # deterministic but shuffled
            random.Random(1337).shuffle(prompts)
            prompts = sorted(set(prompts), key=prompts.index)[:args.per_class]

            out[cls] = prompts

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✓ Wrote {len(out)} classes → {args.out_json}")

if __name__ == "__main__":
    main()

'''
python build_prompts.py --aliases_csv aliases.csv --out_json prompts.json --per_class 12

'''
