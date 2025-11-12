import argparse
import csv
from pathlib import Path

CANON = ["path", "label", "source", "title"]

def pick(candidates, columns):
    cols = [c for c in candidates if c in columns]
    if not cols:
        raise KeyError(f"None of {candidates} found in columns: {list(columns)}")
    return cols[0]

def read_manifest(path, source_name,
                  path_cols=("path", "filepath", "relpath"),
                  label_cols=("label", "category", "class"),
                  title_cols=("title", "image", "id", "name")):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        pcol = pick(path_cols, cols)
        lcol = pick(label_cols, cols)
        tcol = title_cols[0] if any(t in cols for t in title_cols) else None

        for row in r:
            p = (row[pcol] or "").replace("\\", "/")
            if not p.startswith("./"):
                p = "./" + p
            out = {
                "path": p,
                "label": row[lcol],
                "source": source_name,
                "title": row[tcol] if tcol else "",
            }
            rows.append(out)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketchy_csv", required=False)
    ap.add_argument("--quickdraw_csv", required=False)
    ap.add_argument("--out_csv", default="training_manifest.csv")
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop duplicate (path,label) rows.")
    args = ap.parse_args()

    combined = []
    if args.sketchy_csv:
        combined += read_manifest(args.sketchy_csv, "sketchy")
    if args.quickdraw_csv:
        combined += read_manifest(args.quickdraw_csv, "quickdraw")

    if args.dedupe:
        seen = set()
        deduped = []
        for r in combined:
            key = (r["path"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        combined = deduped

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANON)
        w.writeheader()
        w.writerows(combined)

    # Small summary
    n_total = len(combined)
    n_sk = sum(1 for r in combined if r["source"] == "sketchy")
    n_qd = sum(1 for r in combined if r["source"] == "quickdraw")
    n_classes = len(set(r["label"] for r in combined))
    print(f"✓ Wrote {n_total} rows → {args.out_csv}  "
          f"(sketchy={n_sk}, quickdraw={n_qd}, classes={n_classes})")

if __name__ == "__main__":
    main()

'''
python merge_manifests.py --sketchy_csv sketchy_manifest_patched.csv --quickdraw_csv quickdraw_manifest.csv --out_csv training_manifest.csv
'''