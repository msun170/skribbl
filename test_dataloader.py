#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import random
from collections import Counter

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---- Minimal, self-contained CSV dataset (for testing) ----
class CSVImageTextDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        class_index_json: str,
        prompts_json: str = None,
        image_size: int = 224,
        limit: int = None,
        root: str = None,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        # Normalize required cols
        for col in ["path", "label"]:
            if col not in self.manifest.columns:
                raise ValueError(f"Manifest missing required column: {col}")

        # Optional root to prepend to paths (handy on RunPod)
        self.root = Path(root) if root else None

        # Class index
        with open(class_index_json, "r", encoding="utf-8") as f:
            self.class_index = json.load(f)
        self.num_classes = len(self.class_index)

        # Prompts
        if prompts_json and Path(prompts_json).exists():
            with open(prompts_json, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)  # {label: ["prompt1", "prompt2", ...]}
        else:
            self.prompts = {}

        # Basic transforms (CPU-friendly)
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # (Optional) subsample for quick test
        self.rows = self.manifest.to_dict(orient="records")
        if limit is not None:
            self.rows = self.rows[:int(limit)]

        # Quick path check summary (non-fatal)
        self._missing = 0
        for r in self.rows:
            p = self._resolve_path(r["path"])
            if not p.exists():
                self._missing += 1
        if self._missing:
            print(f"⚠️  {self._missing} paths in this subset do not exist on disk (will raise during __getitem__).")

    def _resolve_path(self, relpath: str) -> Path:
        p = Path(relpath)
        if self.root is not None:
            # If manifest paths are relative, resolve under root
            if not p.is_absolute():
                p = self.root / relpath.lstrip("./")
        return p

    def _pick_prompt(self, label: str) -> str:
        cands = self.prompts.get(label)
        if cands:
            return random.choice(cands)
        # default fallback
        return f"a drawing of {label}"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        relpath = r["path"]
        label_str = str(r["label"])
        if label_str not in self.class_index:
            raise KeyError(f"Label '{label_str}' not found in class_index.json.")
        label_id = self.class_index[label_str]

        img_path = self._resolve_path(relpath)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            img = self.tfm(im)

        text = self._pick_prompt(label_str)
        out = {
            "image": img,             # Tensor [3, H, W]
            "text": text,             # str
            "label": label_id,        # int
            "label_str": label_str,   # str
            "path": str(img_path),    # absolute (resolved) for debugging
        }
        # keep any extra manifest columns if you want
        for k in ("source", "title"):
            if k in r:
                out[k] = r[k]
        return out

def collate_fn(batch):
    # images -> stacked float tensor, labels -> long tensor, texts -> list[str]
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts = [b["text"] for b in batch]
    meta = {k: [b.get(k) for b in batch] for k in ("label_str", "path", "source", "title")}
    return imgs, texts, labels, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--class_index", default="class_index.json")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--root", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=64, help="Only load first N rows for quick test")
    ap.add_argument("--batches", type=int, default=2, help="How many batches to iterate for smoke test")
    args = ap.parse_args()

    ds = CSVImageTextDataset(
        manifest_csv=args.manifest,
        class_index_json=args.class_index,
        prompts_json=args.prompts if Path(args.prompts).exists() else None,
        image_size=args.image_size,
        limit=args.limit,
        root=args.root or None,
    )
    print(f"Dataset size (tested subset): {len(ds)}  |  classes: {ds.num_classes}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Iterate a few batches and print shapes + sample rows
    seen_labels = Counter()
    n_images = 0
    for bi, (imgs, texts, labels, meta) in enumerate(dl):
        print(f"\nBatch {bi+1}:")
        print(f"  images:  {tuple(imgs.shape)}  (dtype={imgs.dtype})")
        print(f"  labels:  {tuple(labels.shape)}  min={int(labels.min())} max={int(labels.max())}")
        print(f"  texts[0]: {texts[0]!r}")
        print(f"  label_str[0]: {meta['label_str'][0]!r}")
        print(f"  path[0]: {meta['path'][0]!r}")
        n_images += imgs.shape[0]
        seen_labels.update(meta["label_str"])

        if bi + 1 >= args.batches:
            break

    # Small label coverage summary for the inspected subset
    print("\n— Subset label counts (first few) —")
    for lab, cnt in seen_labels.most_common(10):
        print(f"  {lab:25s} {cnt}")
    print(f"\nOK: iterated {args.batches} batches, {n_images} images total in the smoke test.")

if __name__ == "__main__":
    main()

'''
python test_dataloader.py --manifest training_manifest.csv --class_index class_index.json --prompts prompts.json --batch_size 8 --num_workers 0 --image_size 224 --limit 128 --batches 2

'''