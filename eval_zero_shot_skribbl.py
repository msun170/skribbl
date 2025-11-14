#!/usr/bin/env python3
"""
Evaluate your fine-tuned CLIP in *Skribbl word space*.

It reports:

1) Zero-shot (unconstrained) accuracy:
   - For each test image, we predict over ALL Skribbl words.
   - A prediction is correct if ANY of that class's mapped Skribbl synonyms
     appear in the top-k.

2) Zero-shot (length-constrained) accuracy:
   - For each test image, we pick ONE "canonical" Skribbl word for that class
     (the one with highest semantic similarity in skribbl_semantic_matches.csv).
   - We then mask to ONLY Skribbl words of that exact length (like in-game),
     and check if that canonical word appears in top-k.

We only evaluate images whose dataset class has at least one Skribbl synonym,
and we report the coverage (fraction of test images that are evaluatable).

Assumptions:
- You have:
    linux_training_manifest.csv
    linux_splits.csv
    class_index.json
    skribbl_words.txt
    skribbl_semantic_matches.csv
- dataset_csv.CSVClipDataset works like in your training/eval scripts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from dataset_csv import CSVClipDataset


# ----------------------------
# Helpers for loading metadata
# ----------------------------

def load_skribbl_words(path_txt: str):
    words = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


def load_idx_to_label(class_index_json: str):
    with open(class_index_json, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    # class_index: {label: idx}
    idx_to_label = {v: k for k, v in class_index.items()}
    return idx_to_label


def load_label_to_skribbl_maps(
    matches_csv: str,
    idx_to_label: dict,
    skr_words: list,
    sim_thresh: float = 0.6
):
    """
    Build:

      label_to_all_targets:  label -> set(skr_idx)
      label_to_best_target: label -> single skr_idx with highest similarity

    Using skribbl_semantic_matches.csv.

    We try to detect column names robustly:
      - 'dataset_label' or 'label' or 'target_label' for the dataset class
      - column starting with 'skr' for Skribbl word
      - column containing 'sim' for similarity (cosine)

    Rows with similarity < sim_thresh are ignored.
    """
    df = pd.read_csv(matches_csv)

    # Find dataset label column
    col_label = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("dataset_label", "label", "target_label"):
            col_label = c
            break
    if col_label is None:
        raise ValueError(
            f"Couldn't find dataset label column in {matches_csv}. "
            f"Columns: {list(df.columns)}"
        )

    # Find Skribbl word column
    col_skr = None
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("skr"):
            col_skr = c
            break
    if col_skr is None:
        raise ValueError(
            f"Couldn't find Skribbl word column in {matches_csv}. "
            f"Columns: {list(df.columns)}"
        )

    # Find similarity column
    col_sim = None
    for c in df.columns:
        cl = c.lower()
        if "sim" in cl:
            col_sim = c
            break
    if col_sim is None:
        raise ValueError(
            f"Couldn't find similarity column in {matches_csv}. "
            f"Columns: {list(df.columns)}"
        )

    # Map Skribbl word -> index
    skr_word_to_idx = {w: i for i, w in enumerate(skr_words)}

    # Initialize maps
    label_to_all_targets = {lbl: set() for lbl in idx_to_label.values()}
    label_to_best_target = {lbl: None for lbl in idx_to_label.values()}
    label_to_best_sim = {lbl: -1.0 for lbl in idx_to_label.values()}

    for _, row in df.iterrows():
        lbl = str(row[col_label])
        skr = str(row[col_skr])
        sim = float(row[col_sim])

        if sim < sim_thresh:
            continue
        if skr not in skr_word_to_idx:
            continue
        if lbl not in label_to_all_targets:
            # likely an ImageNet or pruned label not in your final 330
            continue

        skr_idx = skr_word_to_idx[skr]
        label_to_all_targets[lbl].add(skr_idx)

        # Track best (highest similarity) Skribbl word per label
        if sim > label_to_best_sim[lbl]:
            label_to_best_sim[lbl] = sim
            label_to_best_target[lbl] = skr_idx

    # Remove labels that ended up with no targets at all
    return label_to_all_targets, label_to_best_target


# ----------------------------
# Model + data helpers
# ----------------------------

def build_skribbl_text_features(model, tokenizer, skr_words, device):
    """
    Compute one text embedding per Skribbl word.
    """
    batch_size = 512
    feats = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(skr_words), batch_size),
                      desc="Encoding Skribbl words"):
            chunk = skr_words[i : i + batch_size]
            tokens = tokenizer(chunk).to(device)
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                emb = model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.cpu())

    text_features = torch.cat(feats, dim=0)  # (N_skr, D)
    return text_features.to(device)


def collate_eval(batch):
    """
    CSVClipDataset items:
      (image, label, text) or (image, label, text, path)

    We only need image + label.
    """
    imgs, ys = [], []
    for item in batch:
        if len(item) == 3:
            img, y, _txt = item
        elif len(item) == 4:
            img, y, _txt, _path = item
        else:
            raise ValueError(f"Unexpected item length {len(item)}")
        imgs.append(img)
        ys.append(int(y if not torch.is_tensor(y) else y.item()))
    return torch.stack(imgs, 0), torch.tensor(ys, dtype=torch.long)


def load_finetuned_model(model_name, ckpt_path, device):
    """
    Load your fine-tuned CLIP checkpoint into an open_clip model skeleton.
    """
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        precision="fp16" if device == "cuda" else "fp32",
    )
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt
        )
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    return model


# ----------------------------
# Evaluation core
# ----------------------------

def eval_zero_shot_skribbl(
    model,
    skr_text_feats,
    skr_lengths,
    loader,
    idx_to_label,
    label_to_all_targets,
    label_to_best_target,
    device,
    topk=5,
):
    """
    Evaluate zero-shot in Skribbl word space, returning:

    - top1_uc, topk_uc : unconstrained
    - top1_len, topk_len : length-constrained (canonical word per label)
    - n_covered_uc : #images with at least 1 Skribbl synonym
    - n_covered_len : #images with a canonical synonym (usually same as uc)
    """
    model.eval()

    top1_uc = topk_uc = 0
    top1_len = topk_len = 0
    covered_uc = covered_len = 0

    with torch.no_grad():
        for images, ys in tqdm(loader, desc="Eval zero-shot (Skribbl space)"):
            images = images.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            # Encode images
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                img_feats = model.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits = img_feats @ skr_text_feats.t()  # (B, N_skr)

            # Precompute unconstrained top-k
            _, topk_idx = logits.topk(topk, dim=1)  # (B, topk)

            for i in range(images.size(0)):
                class_idx = int(ys[i].item())
                label = idx_to_label[class_idx]

                # ---- Unconstrained (any Skribbl synonym) ----
                tgt_set = label_to_all_targets.get(label, set())
                if tgt_set:
                    covered_uc += 1
                    preds = topk_idx[i].tolist()
                    if preds[0] in tgt_set:
                        top1_uc += 1
                    if any(p in tgt_set for p in preds):
                        topk_uc += 1

                # ---- Length-constrained (canonical Skribbl word) ----
                canonical_idx = label_to_best_target.get(label, None)
                if canonical_idx is not None:
                    covered_len += 1
                    target_len = int(skr_lengths[canonical_idx].item())

                    # mask to length == target_len
                    len_mask = (skr_lengths == target_len)  # (N_skr,)
                    # large negative for masked-out
                    masked_logits = logits[i].clone()
                    masked_logits[~len_mask] = -1e9
                    _, topk_idx_len = masked_logits.topk(topk, dim=0)

                    preds_len = topk_idx_len.tolist()
                    if preds_len[0] == canonical_idx:
                        top1_len += 1
                    if canonical_idx in preds_len:
                        topk_len += 1

    def safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    results = {
        "top1_unconstrained": safe_div(top1_uc, covered_uc),
        "topk_unconstrained": safe_div(topk_uc, covered_uc),
        "top1_lenconstrained": safe_div(top1_len, covered_len),
        "topk_lenconstrained": safe_div(topk_len, covered_len),
        "covered_unconstrained": covered_uc,
        "covered_lenconstrained": covered_len,
    }
    return results


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--skr_words", default="skribbl_words.txt")
    ap.add_argument("--matches", default="skribbl_semantic_matches.csv")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sim_thresh", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset: test split only
    ds_test = CSVClipDataset(
        manifest_csv=args.manifest,
        splits_csv=args.splits,
        class_index_json=args.class_index,
        prompts_json=None,     # we don't need prompts for eval
        split="test",
        img_size=args.img_size,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_eval,
    )
    print(f"Test set size: {len(ds_test)}")

    # Index ↔ label
    idx_to_label = load_idx_to_label(args.class_index)

    # Skribbl vocabulary
    skr_words = load_skribbl_words(args.skr_words)
    print(f"Skribbl vocabulary size: {len(skr_words)}")

    # Label → Skribbl mappings
    label_to_all_targets, label_to_best_target = load_label_to_skribbl_maps(
        args.matches,
        idx_to_label,
        skr_words,
        sim_thresh=args.sim_thresh,
    )

    n_labels_any = sum(1 for v in label_to_all_targets.values() if len(v) > 0)
    n_labels_canon = sum(1 for v in label_to_best_target.values() if v is not None)
    print(
        f"Labels with ≥1 Skribbl synonym: {n_labels_any}/{len(idx_to_label)}; "
        f"labels with canonical synonym: {n_labels_canon}/{len(idx_to_label)}"
    )

    # Model + tokenizer
    tokenizer = open_clip.get_tokenizer(args.model)
    model = load_finetuned_model(args.model, args.ckpt, device)

    # Skribbl text features
    skr_text_feats = build_skribbl_text_features(model, tokenizer, skr_words, device)
    # Vector of word lengths for length-constraint masking
    skr_lengths = torch.tensor([len(w) for w in skr_words], device=device, dtype=torch.long)

    # Run evaluation
    results = eval_zero_shot_skribbl(
        model,
        skr_text_feats,
        skr_lengths,
        loader_test,
        idx_to_label,
        label_to_all_targets,
        label_to_best_target,
        device,
        topk=args.topk,
    )

    cov_uc = results["covered_unconstrained"] / len(ds_test) if len(ds_test) > 0 else 0.0
    cov_len = results["covered_lenconstrained"] / len(ds_test) if len(ds_test) > 0 else 0.0

    print("\n=== Zero-shot Skribbl-space results (test split) ===")
    print(f"Coverage (unconstrained): {results['covered_unconstrained']} / {len(ds_test)} "
          f"({cov_uc:.3%})")
    print(f"  top1 (unconstrained): {results['top1_unconstrained']:.3f}")
    print(f"  top{args.topk} (unconstrained): {results['topk_unconstrained']:.3f}")

    print(f"\nCoverage (length-constrained): {results['covered_lenconstrained']} / {len(ds_test)} "
          f"({cov_len:.3%})")
    print(f"  top1 (length-constrained): {results['top1_lenconstrained']:.3f}")
    print(f"  top{args.topk} (length-constrained): {results['topk_lenconstrained']:.3f}")


if __name__ == "__main__":
    main()
