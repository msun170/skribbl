import argparse, json
import pandas as pd

def pick(colnames, columns):
    for c in colnames:
        if c in columns: return c
    raise KeyError(f"Need one of {colnames}, but have {list(columns)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--out_json", default="class_index.json")
    ap.add_argument("--sort", choices=["alpha","freq"], default="alpha",
                    help="alpha: alphabetical by label; freq: most frequent first")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    label_col = pick(["label","category","class"], df.columns)

    if args.sort == "alpha":
        labels = sorted(df[label_col].astype(str).str.strip().unique())
    else:
        counts = df.groupby(label_col)[label_col].count().sort_values(ascending=False)
        labels = list(counts.index.astype(str))

    class_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print(f"✓ Wrote {len(labels)} classes -> {args.out_json}")

if __name__ == "__main__":
    main()

'''
python build_class_index.py --manifest training_manifest.csv --out_json class_index.json --sort alpha
'''