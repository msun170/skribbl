# match_strength_plots.py
# Usage:
#   python match_strength_plots.py skribbl_semantic_matches.csv 0.60
# If you omit args, it defaults to the file name below and threshold=0.60

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- inputs ----
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("skribbl_semantic_matches.csv")
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60

# ---- load & normalize columns ----
df = pd.read_csv(csv_path)

# make a lowercased map to find columns flexibly
lower_map = {c.lower(): c for c in df.columns}

def pick(colnames):
    for name in colnames:
        if name in lower_map:
            return lower_map[name]
    raise KeyError(f"None of {colnames} found in columns: {list(df.columns)}")

sim_col = "cosine_similarity"
dataset_col = "best_match_dataset"
# Optional helpful columns if present, but not required:
skribbl_col = lower_map.get("skribbl_word", None) or lower_map.get("query", None)
target_word_col = lower_map.get("target_word", None) or lower_map.get("match", None)

# keep only sensible similarities
df = df[(df[sim_col] >= 0.0) & (df[sim_col] <= 1.0)].copy()

# ---- quick textual summary (printed to console) ----
print("=== Match Strength Summary ===")
print(f"Rows: {len(df)}   (file: {csv_path})")
print(f"Threshold: {threshold:.2f}")
print(df[sim_col].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
print()

# ---- Overall histogram ----
bins = np.linspace(0.0, 1.0, 41)  # 40 bins across [0,1]

plt.figure()
plt.hist(df[sim_col].values, bins=bins)
plt.axvline(threshold)
plt.title(f"Match Strength Distribution (All datasets)\nN={len(df)} | threshold={threshold:.2f}")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
out1 = Path("similarity_hist_overall.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote {out1}")

# ---- Overall CDF ----
vals = np.sort(df[sim_col].values)
y = np.arange(1, len(vals) + 1) / len(vals)

plt.figure()
plt.plot(vals, y)
plt.axvline(threshold)
plt.title(f"CDF of Match Strength (All datasets)\nN={len(vals)} | threshold={threshold:.2f}")
plt.xlabel("Cosine similarity")
plt.ylabel("Cumulative fraction ≤ x")
out2 = Path("similarity_cdf_overall.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote {out2}")

# ---- Per-dataset histograms (one image per dataset) ----
if dataset_col in df.columns:
    for name, dsub in df.groupby(dataset_col):
        plt.figure()
        plt.hist(dsub[sim_col].values, bins=bins)
        plt.axvline(threshold)
        plt.title(f"Match Strength: {name}\nN={len(dsub)} | threshold={threshold:.2f}")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        outp = Path(f"similarity_hist_{str(name).lower()}.png")
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {outp}")

print("\nDone.")
