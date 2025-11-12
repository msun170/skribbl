# convert_quickdraw_ndjson_to_png.py
import argparse, json, os, re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageDraw

def truthy(x):
    s = str(x).strip().lower()
    return s not in ("", "0", "false", "no", "none")

def norm_label(s: str) -> str:
    # normalize labels for matching against file stems
    return re.sub(r"\s+", " ", s.strip().lower())

def load_matched_quickdraw_classes(matches_csv: Path, min_cosine=0.6):
    import csv
    wanted = set()
    with open(matches_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("best_match_dataset", "").lower() != "quickdraw":
                continue
            if not truthy(row.get("is_match", "")):
                continue
            try:
                if float(row.get("cosine_similarity", "0")) < min_cosine:
                    continue
            except:
                pass
            # prefer normalized; fall back to raw
            lbl = row.get("best_match_norm") or row.get("best_match_raw") or ""
            if lbl:
                wanted.add(norm_label(lbl))
    return wanted

def index_ndjson_files(ndjson_dir: Path):
    """
    Return dict: normalized_stem -> Path(ndjson)
    Handles filenames like 'airplane.ndjson' or 'the eiffel tower.ndjson'
    """
    mapping = {}
    for p in ndjson_dir.glob("*.ndjson"):
        mapping[norm_label(p.stem)] = p
    return mapping

def draw_sample(sample, size=256, line_width=4):
    """
    Render a single QuickDraw sample (JSON object) to a PIL Image.
    Works with 'drawing' as list of strokes:
      stroke = [[x...],[y...]]  or [[x...],[y...],[t...]]
    We rescale to fit a 0..255 box with small margins.
    """
    drawing = sample["drawing"]
    # Flatten points for bounds
    xs, ys = [], []
    for s in drawing:
        x = s[0]; y = s[1]
        xs.extend(x); ys.extend(y)
    if not xs or not ys:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)
    # Fit to size with 5% padding
    pad = 0.05
    scale = (1.0 - 2*pad) * size / max(w, h)
    xoff = (size - scale * (xmin + xmax)) / 2.0
    yoff = (size - scale * (ymin + ymax)) / 2.0

    img = Image.new("L", (size, size), color=255)  # white
    d = ImageDraw.Draw(img)
    # Draw each stroke
    for s in drawing:
        x = s[0]; y = s[1]
        if len(x) < 2: 
            continue
        pts = [(xoff + xi*scale, yoff + yi*scale) for xi, yi in zip(x, y)]
        # Draw as polyline; join/cap round by using many small segments
        d.line(pts, fill=0, width=line_width, joint="curve")
    return img.convert("RGB")  # RGB is friendlier for downstream tooling

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches_csv", required=True, help="skribbl_semantic_matches.csv")
    ap.add_argument("--ndjson_dir", required=True, help="Directory containing QuickDraw .ndjson files")
    ap.add_argument("--out_root", required=True, help="Output root for PNGs, e.g. ./quickdraw_png256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--per_class", type=int, default=3000)
    ap.add_argument("--min_cosine", type=float, default=0.6)
    ap.add_argument("--only_recognized", action="store_true", help="Keep only samples where recognized==true")
    ap.add_argument("--line_width", type=int, default=4)
    args = ap.parse_args()

    matches_csv = Path(args.matches_csv)
    ndjson_dir  = Path(args.ndjson_dir)
    out_root    = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = load_matched_quickdraw_classes(matches_csv, min_cosine=args.min_cosine)
    stem_map = index_ndjson_files(ndjson_dir)

    missing_files = []
    written = 0
    per_class_counts = defaultdict(int)

    for lbl in tqdm(sorted(wanted), desc="Classes"):
        ndjson_path = stem_map.get(lbl)
        if ndjson_path is None:
            missing_files.append(lbl)
            continue
        cls_dir = out_root / ndjson_path.stem  # keep original stem for folder
        cls_dir.mkdir(parents=True, exist_ok=True)

        with open(ndjson_path, "r", encoding="utf-8") as f:
            for line in f:
                if per_class_counts[lbl] >= args.per_class:
                    break
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if args.only_recognized and not truthy(sample.get("recognized", True)):
                    continue
                img = draw_sample(sample, size=args.size, line_width=args.line_width)
                if img is None:
                    continue
                key_id = str(sample.get("key_id", "")).strip()
                if not key_id:
                    # fallback: use a running count, but prefer key_id if present
                    key_id = f"idx_{per_class_counts[lbl]}"
                out_path = cls_dir / f"{key_id}.png"
                if out_path.exists():
                    # already done
                    per_class_counts[lbl] += 1
                    continue
                img.save(out_path)
                written += 1
                per_class_counts[lbl] += 1

    print(f"\nDone. Wrote {written} PNGs to {out_root}")
    if missing_files:
        print("Missing .ndjson for labels (normalized):")
        for s in sorted(missing_files):
            print("  -", s)

if __name__ == "__main__":
    main()

'''
python convert_quickdraw_ndjson_to_png.py --matches_csv skribbl_semantic_matches.csv --ndjson_dir quickdraw_ndjson --out_root quickdraw_png256 --per_class 3000 --size 256 --only_recognized --line_width 4


'''