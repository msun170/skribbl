#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import open_clip

from dataset_csv import CSVClipDataset  # your dataset

# ---------------- Collate that tolerates (img, label, text) OR (img, label, text, path)
def collate(batch):
    """
    Accept items of shape:
      (image, label, text)            OR
      (image, label, text, path)
    Returns: (images_tensor, texts_list, labels_tensor)
    """
    images, labels, texts = [], [], []
    for item in batch:
        if len(item) == 3:
            img, lbl, txt = item
        elif len(item) == 4:
            img, lbl, txt, _path = item  # path is metadata; discard for training
        else:
            raise ValueError(f"Unexpected item length {len(item)} (expected 3 or 4)")
        images.append(img)
        labels.append(lbl)
        texts.append(txt)
    return torch.stack(images, 0), texts, torch.stack(labels, 0)


def build_loaders(args):
    train_ds = CSVClipDataset(
        manifest_csv=args.manifest,
        splits_csv=args.splits,
        class_index_json=args.class_index,
        prompts_json=args.prompts,
        split="train",
        img_size=args.img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=collate,
        drop_last=True,
    )

    val_loader = None
    # Try to build a val loader if split exists
    try:
        val_ds = CSVClipDataset(
            manifest_csv=args.manifest,
            splits_csv=args.splits,
            class_index_json=args.class_index,
            prompts_json=args.prompts,
            split="val",
            img_size=args.img_size,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=(args.workers > 0),
                collate_fn=collate,
                drop_last=False,
            )
    except Exception:
        pass

    return train_loader, val_loader


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
    # Similarity
    logits_per_image = logit_scale.exp() * image_features @ text_features.t()
    logits_per_text  = logits_per_image.t()
    batch_size = image_features.size(0)
    targets = torch.arange(batch_size, device=image_features.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t = nn.functional.cross_entropy(logits_per_text,  targets)
    return (loss_i + loss_t) / 2, logits_per_image


def evaluate(model, tokenizer, val_loader, device):
    model.eval()
    n, running_loss = 0, 0.0
    with torch.no_grad(), torch.amp.autocast('cuda' if device=='cuda' else 'cpu'):
        for images, texts, _ in tqdm(val_loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            tokenized = tokenizer(texts).to(device, non_blocking=True)

            image_features = model.encode_image(images)
            text_features  = model.encode_text(tokenized)
            loss, _ = clip_loss(image_features, text_features, model.logit_scale)

            bs = images.size(0)
            running_loss += loss.item() * bs
            n += bs
    return running_loss / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", default=None)
    p.add_argument("--class_index", default=None)
    p.add_argument("--prompts", default=None)

    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)

    p.add_argument("--out_dir", default="checkpoints")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build data
    train_loader, val_loader = build_loaders(args)

    # Build model + tokenizer
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler (new API)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        seen = 0

        for images, texts, _labels in pbar:
            images = images.to(device, non_blocking=True)
            tokenized = tokenizer(texts).to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                image_features = model.encode_image(images)
                text_features  = model.encode_text(tokenized)
                loss, _ = clip_loss(image_features, text_features, model.logit_scale)

            scaler.scale(loss).to(device)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            bs = images.size(0)
            running += loss.item() * bs
            seen += bs
            global_step += 1

            pbar.set_postfix(loss=f"{running/seen:.4f}", bs=bs)

        # End of epoch: optional val
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, tokenizer, val_loader, device)

        # Save checkpoint
        ckpt_path = Path(args.out_dir) / f"clip_{args.model.replace('/','-')}_e{epoch+1}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
                "val_loss": val_loss,
            },
            ckpt_path,
        )

        if val_loader is not None:
            print(f"[epoch {epoch+1}] train_loss={running/seen:.4f}  val_loss={val_loss:.4f}  saved={ckpt_path}")
        else:
            print(f"[epoch {epoch+1}] train_loss={running/seen:.4f}  saved={ckpt_path}")


if __name__ == "__main__":
    main()
