#!/usr/bin/env python3
"""
eval_clip_length_constrained.py

Evaluates a CLIP model on your manifest/splits with TWO protocols:
  1) Standard: rank over all classes (Top-1 / Top-5)
  2) Length-constrained: rank only among classes whose canonical label has
     the same character count (letters-only, no spaces/punct) as the GT label
     (Top-1 / Top-5). This simulates the game's "word length" hint.

Assumptions:
- You have: training_manifest.csv, splits.csv, class_index.json, prompts.json
- You can run on CPU (no GPU required) for pipeline sanity checks.
- Works with either open_clip or HuggingFace CLIP (auto-detected via --backend).

Example:
  python eval_clip_length_constrained.py \
      --manifest training_manifest.csv \
      --splits splits.csv \
      --class_index class_index.json \
      --prompts prompts.json \
      --split test \
      --backend open_clip \
      --model ViT-B-32 \
      --pretrained laion2b_s34b_b79k \
      --batch_size 64 \
      --device cpu
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd

# -------------------------------
# Utilities
# -------------------------------
def norm_label_for_length(s: str) -> str:
    """Normalize for letter-count: remove spaces/punct, lowercase."""
    s = s.lower()
    s = re.sub(r"[^a-z]", "", s)  # keep letters a-z only
    return s

def letter_len(s: str) -> int:
    return len(norm_label_for_length(s))

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def average_l2_normalized(embs: torch.Tensor) -> torch.Tensor:
    # embs: [N, D]
    embs = torch.nn.functional.normalize(embs, dim=-1)
    avg = embs.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(avg, dim=-1)  # [1, D]

# -------------------------------
# CLIP backends
# -------------------------------
def build_openclip(model_name: str, pretrained: str, device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Ensure image transform matches your pngs (preprocess is fine; add center-crop safety)
    # Note: open_clip preprocess already includes Resize/CenterCrop/ToTensor/Normalize
    return model, tokenizer, preprocess

def build_hfclip(model_name_or_path: str, device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name_or_path)
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    model.to(device)

    # For HF we’ll wrap minimal API for parity
    def tokenize(texts: List[str]) -> Dict[str, torch.Tensor]:
        tok = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        return {k: v for k, v in tok.items() if k in ("input_ids", "attention_mask")}

    # simple image transform that matches processor; we’ll use processor for images anyway
    def preprocess_pil(pil: Image.Image) -> Dict[str, torch.Tensor]:
        pix = processor(images=pil, return_tensors="pt")
        return pix

    return model, tokenize, preprocess_pil, processor

# -------------------------------
# Data helpers
# -------------------------------
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv: str, split_csv: str, which_split: str,
                 class_index: Dict[str, int], img_transform):
        self.df = pd.read_csv(manifest_csv)
        splits = pd.read_csv(split_csv)

        # Expect columns: path,label in manifest; id or path join key in splits
        # We’ll join on path, which is unique in your pipeline.
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("manifest must contain columns: path,label,...")

        if "path" not in splits.columns or "split" not in splits.columns:
            raise ValueError("splits must contain columns: path,split,...")

        merged = self.df.merge(splits[["path", "split"]], on="path", how="inner")
        merged = merged[merged["split"].str.lower() == which_split.lower()].copy()

        # Keep only rows whose label is in class_index
        merged = merged[merged["label"].isin(class_index.keys())].reset_index(drop=True)

        self.df = merged
        self.class_index = class_index
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r["path"]
        label = r["label"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback: try without leading "./"
            p = path[2:] if path.startswith("./") else path
            img = Image.open(p).convert("RGB")
        if callable(self.img_transform):
            img = self.img_transform(img)
        y = self.class_index[label]
        return img, label, path, y

# -------------------------------
# Build class text embeddings
# -------------------------------
def build_class_text_embeds_openclip(model, tokenizer, prompts_json, label_list, device):
    texts = []
    idx_of_class = []  # map back
    for cls in label_list:
        variants = prompts_json.get(cls, [f"a drawing of a {cls}", f"a sketch of a {cls}"])
        # tokenize all variants
        tok = tokenizer(variants)
        with torch.no_grad():
            # open_clip: model.encode_text expects tokenized ints
            txt = model.encode_text(tok.to(device))
        cls_embed = average_l2_normalized(txt)  # [1, D]
        texts.append(cls_embed)
        idx_of_class.append(cls)
    return torch.cat(texts, dim=0), idx_of_class  # [C, D], list of class names

def build_class_text_embeds_hf(model, processor, prompts_json, label_list, device):
    embs = []
    idx_of_class = []
    for cls in label_list:
        variants = prompts_json.get(cls, [f"a drawing of a {cls}", f"a sketch of a {cls}"])
        # Encode each variant and average
        var_embs = []
        for v in variants:
            with torch.no_grad():
                tok = processor(text=[v], return_tensors="pt", padding=True, truncation=True).to(device)
                txt = model.get_text_features(**tok)  # [1, D]
                txt = torch.nn.functional.normalize(txt, dim=-1)
            var_embs.append(txt)
        cls_embed = torch.stack(var_embs, dim=0).mean(dim=0)
        cls_embed = torch.nn.functional.normalize(cls_embed, dim=-1)
        embs.append(cls_embed)
        idx_of_class.append(cls)
    return torch.cat(embs, dim=0), idx_of_class

# -------------------------------
# Evaluation
# -------------------------------
@torch.no_grad()
def eval_loop(
    model,
    dataloader,
    device,
    class_names: List[str],
    class_embeds: torch.Tensor,  # [C, D] normalized
    backend: str,
    hf_processor=None
):
    """
    Returns:
      metrics dict and per-sample rows (for CSV).
    """
    # Precompute lengths (letters only) per class
    class_len = [letter_len(c) for c in class_names]
    C = len(class_names)

    # Move class embeds once
    class_embeds = class_embeds.to(device)  # [C, D]

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_len_top1 = 0
    correct_len_top5 = 0

    per_rows = []  # for CSV

    for batch in dataloader:
        imgs, labels, paths, y = batch  # labels are str, y is int (not used here)
        b = imgs.size(0)
        total += b

        imgs = imgs.to(device)

        # Encode images
        if backend == "open_clip":
            img_feats = model.encode_image(imgs)
        else:  # hf
            # If using HF processor, it expects dicts; but for speed we passed tensors via Dataset transforms.
            img_feats = model.get_image_features(pixel_values=imgs)

        img_feats = torch.nn.functional.normalize(img_feats, dim=-1)  # [B, D]

        # Cosine sim ~ dot product of normalized vectors
        logits = img_feats @ class_embeds.T  # [B, C]

        # Standard Top-k
        topk = min(5, C)
        topk_vals, topk_idx = logits.topk(topk, dim=1)  # [B, K]

        # Length-constrained Top-k
        # Build mask per sample: keep classes with matching letter length to GT label
        gt_lens = [letter_len(l) for l in labels]
        mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, C]
        for i, L in enumerate(gt_lens):
            # mark candidates matching that length
            # (simple equal-length rule; you can relax to +/-1 if desired)
            ok = [j for j, cl in enumerate(class_len) if cl == L]
            if ok:
                mask[i, torch.tensor(ok, device=mask.device)] = True
            else:
                # If no candidates of same length, fall back to all (avoid empty)
                mask[i, :] = True

        masked_logits = logits.masked_fill(~mask, -1e9)  # large negative where disallowed

        len_topk_vals, len_topk_idx = masked_logits.topk(topk, dim=1)  # [B, K]

        # Compute correctness
        for i in range(b):
            gt = labels[i]
            gt_idx = class_names.index(gt) if gt in class_names else None

            # Unconstrained checks
            preds = [class_names[j] for j in topk_idx[i].tolist()]
            hit_top1 = (preds[0] == gt)
            hit_top5 = (gt in preds)

            # Len-constrained checks
            lpreds = [class_names[j] for j in len_topk_idx[i].tolist()]
            l_hit_top1 = (lpreds[0] == gt)
            l_hit_top5 = (gt in lpreds)

            correct_top1 += int(hit_top1)
            correct_top5 += int(hit_top5)
            correct_len_top1 += int(l_hit_top1)
            correct_len_top5 += int(l_hit_top5)

            per_rows.append({
                "path": paths[i],
                "label": gt,
                "pred_top1": preds[0],
                "preds_top5": "|".join(preds),
                "len_pred_top1": lpreds[0],
                "len_preds_top5": "|".join(lpreds),
                "gt_len": letter_len(gt),
                "top1_correct": int(hit_top1),
                "top5_correct": int(hit_top5),
                "len_top1_correct": int(l_hit_top1),
                "len_top5_correct": int(l_hit_top5),
            })

    metrics = {
        "num_samples": int(total),
        "top1": correct_top1 / max(1, total),
        "top5": correct_top5 / max(1, total),
        "len_top1": correct_len_top1 / max(1, total),
        "len_top5": correct_len_top5 / max(1, total),
    }
    return metrics, per_rows

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--split", default="test")

    ap.add_argument("--backend", choices=["open_clip", "hf"], default="open_clip")
    ap.add_argument("--model", default="ViT-B-32", help="open_clip model name OR HF model name/path")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained tag (ignored for HF)")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--out_metrics", default="eval_length_metrics.json")
    ap.add_argument("--out_predictions", default="eval_length_predictions.csv")
    args = ap.parse_args()

    device = args.device

    # Load label space and prompts
    class_index = load_json(args.class_index)  # {label: idx}
    # Ensure deterministic order for class list
    labels_sorted = sorted(class_index.keys(), key=lambda s: class_index[s])

    prompts = load_json(args.prompts)  # {label: [prompt1, prompt2, ...]}

    # Build transforms and model
    if args.backend == "open_clip":
        model, tokenizer, preprocess = build_openclip(args.model, args.pretrained, device)
        model.eval()
        # Build class text embeddings (avg over prompt variants)
        class_embeds, class_names = build_class_text_embeds_openclip(
            model, tokenizer, prompts, labels_sorted, device
        )
        # Dataset transform (open_clip preprocess is fine)
        img_transform = preprocess
        hf_processor = None
    else:
        model, tokenize, preprocess_pil, hfproc = build_hfclip(args.model, device)
        model.eval()
        # Build class text embeddings
        class_embeds, class_names = build_class_text_embeds_hf(
            model, hfproc, prompts, labels_sorted, device
        )
        # For HF path we’ll use a transforms pipeline to return normalized tensors compatible with processor
        # Match CLIP’s 224 short-side defaults (processor will handle if needed)
        img_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        hf_processor = hfproc

    # Dataset / Loader
    ds = ManifestDataset(args.manifest, args.splits, args.split, class_index, img_transform)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.startswith("cuda"))
    )

    # Eval
    metrics, rows = eval_loop(
        model=model,
        dataloader=dl,
        device=device,
        class_names=class_names,
        class_embeds=class_embeds,
        backend=args.backend,
    )

    # Save
    Path(args.out_predictions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_predictions, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "path","label","pred_top1","preds_top5","len_pred_top1","len_preds_top5",
            "gt_len","top1_correct","top5_correct","len_top1_correct","len_top5_correct"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
