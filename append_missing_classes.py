import csv
from pathlib import Path

# Classes we know exist on disk but were missing from the manifest
MISSING_CLASSES = [
    "teddy_bear",
    "hot-air_balloon",
    "sea_turtle",
    "car_(sedan)",
    "wine_bottle",
]

sketchy_root = Path("sketchy_png256")
manifest_path = Path("sketchy_manifest.csv")
out_path = Path("sketchy_manifest_patched.csv")

rows = []
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Normalize: use consistent column names if they differ slightly
normalized_fieldnames = [c.lower() for c in fieldnames]
def has_col(name):
    return name.lower() in normalized_fieldnames

# Prepare a template row so new entries match existing columns exactly
template = {k: "" for k in fieldnames}

added = 0
for cls in MISSING_CLASSES:
    folder = sketchy_root / cls
    if not folder.exists():
        print(f"⚠️ Missing folder for {cls}, skipping.")
        continue

    for img in folder.glob("*.png"):
        rel_path = f"./{sketchy_root.name}/{cls}/{img.name}"
        new_row = template.copy()
        # Fill the obvious fields
        if has_col("path"):
            new_row["path"] = rel_path
        if has_col("title"):
            new_row["title"] = img.stem
        if has_col("category"):
            new_row["category"] = cls
        elif has_col("label"):
            new_row["label"] = cls
        added += 1
        rows.append(new_row)

print(f"✓ Added {added} new rows for missing classes")

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Wrote patched manifest → {out_path}")

'''
python append_missing_classes.py
'''
