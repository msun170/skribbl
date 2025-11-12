import os
import re
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
PATH_SKRIBBL = "skribbl_words.txt"                 # your uploaded list
PATH_IMAGENET = "imagenet classes.txt"
PATH_QUICKDRAW = "quickdraw_classes.txt"
PATH_SKETCHY = "sketchy database.txt"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # light, fast, solid
SIM_THRESHOLD = 0.60                                     # cosine similarity cutoff
TOP_K = 1                                                # keep best match per word

# ----------------------------
# HELPERS
# ----------------------------
def normalize(s: str) -> str:
    # lower, strip, collapse whitespace, strip accents, keep basic punctuation
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

def load_wordlist(path: str, source_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f if line.strip()]
    df = pd.DataFrame({
        "raw": raw,
        "norm": [normalize(x) for x in raw],
        "source": source_name
    })
    # Drop duplicates on normalized form, keep first occurrence
    df = df.drop_duplicates(subset=["norm"]).reset_index(drop=True)
    return df

def embed_texts(model, texts, batch_size=256):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", ncols=80):
        embs.append(model.encode(texts[i:i+batch_size], show_progress_bar=False, normalize_embeddings=True))
    return np.vstack(embs) if embs else np.zeros((0, 384), dtype=np.float32)

# ----------------------------
# LOAD DATA
# ----------------------------
skribbl = load_wordlist(PATH_SKRIBBL, "skribbl")
imagenet = load_wordlist(PATH_IMAGENET, "imagenet")
quickdraw = load_wordlist(PATH_QUICKDRAW, "quickdraw")
sketchy = load_wordlist(PATH_SKETCHY, "sketchy")

targets = pd.concat([imagenet, quickdraw, sketchy], ignore_index=True)
targets["dataset"] = targets["source"]  # alias

print(f"Skribbl words: {len(skribbl)}")
print(f"Targets total: {len(targets)} "
      f"(ImageNet={len(imagenet)}, QuickDraw={len(quickdraw)}, Sketchy={len(sketchy)})")

# ----------------------------
# EMBEDDINGS
# ----------------------------
model = SentenceTransformer(EMBED_MODEL)
q_emb = embed_texts(model, skribbl["norm"].tolist())     # (Nq, D), L2-normalized
t_emb = embed_texts(model, targets["norm"].tolist())     # (Nt, D), L2-normalized

# ----------------------------
# NEAREST MATCH (cosine)
# ----------------------------
# With normalized embeddings, cosine similarity = dot product
# We’ll chunk to avoid huge memory spikes if lists are large.
def batched_topk_dot(A, B, k=1, chunk=2000):
    # A: (Nq, D), B: (Nt, D) normalized; returns top-k per row in A
    sims_list, idx_list = [], []
    for i in tqdm(range(0, A.shape[0], chunk), desc="Searching", ncols=80):
        a = A[i:i+chunk]                  # (c, D)
        sim = a @ B.T                     # (c, Nt)
        # get top-k indices and values along axis=1
        top_idx = np.argpartition(-sim, kth=range(k), axis=1)[:, :k]
        # refine exact sort within top-k
        row_sorted = np.take_along_axis(sim, top_idx, axis=1)
        order = np.argsort(-row_sorted, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        row_sorted = np.take_along_axis(row_sorted, order, axis=1)
        sims_list.append(row_sorted)
        idx_list.append(top_idx)
    sims = np.vstack(sims_list)
    idxs = np.vstack(idx_list)
    return sims, idxs

top_sims, top_idxs = batched_topk_dot(q_emb, t_emb, k=TOP_K)

best_idx = top_idxs[:, 0]
best_sim = top_sims[:, 0]

# ----------------------------
# BUILD RESULTS
# ----------------------------
best_matches = targets.iloc[best_idx].reset_index(drop=True)
out = pd.DataFrame({
    "skribbl_raw": skribbl["raw"],
    "skribbl_norm": skribbl["norm"],
    "best_match_raw": best_matches["raw"],
    "best_match_norm": best_matches["norm"],
    "best_match_dataset": best_matches["dataset"],
    "cosine_similarity": best_sim
})

# Flag matches above threshold
out["is_match"] = out["cosine_similarity"] >= SIM_THRESHOLD

# Summary counts
total = len(out)
matched = int(out["is_match"].sum())
coverage = matched / total if total else 0.0

by_dataset = (
    out.loc[out["is_match"]]
    .groupby("best_match_dataset")["is_match"]
    .count()
    .reindex(["imagenet", "quickdraw", "sketchy"], fill_value=0)
    .to_dict()
)

print("\n=== Summary ===")
print(f"Threshold: {SIM_THRESHOLD:.2f}")
print(f"Total Skribbl words: {total}")
print(f"Matched (any dataset): {matched}  ({coverage:.1%})")
print("Matched by dataset:", by_dataset)

# Save detailed CSV
csv_path = "skribbl_semantic_matches.csv"
out.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\nWrote details → {csv_path}")
