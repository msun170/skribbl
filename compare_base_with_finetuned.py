#!/usr/bin/env python3
"""
Compare base OpenAI CLIP vs your fine-tuned CLIP checkpoint
on the same test split (unconstrained top-1/top-5 accuracy).

Usage (example):

  python compare_base_vs_finetuned.py \
    --manifest linux_training_manifest.csv \
    --splits linux_splits.csv \
    --class_index class_index.json \
    --prompts prompts.json \
    --model ViT-B-32 \
    --ckpt checkpoints/clip_ViT-B-32_e5.pt \
    --batch_size 128 \
    --num_workers 8 \
    --img_size 224
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from dataset_csv import CSVClipDataset


def collate_eval(batch):
    """
    Batch items from CSVClipDataset:
      (image, label, text) or (image, label, text, path)
    Return:
      images: (B, C, H, W)
      labels: (B,) long
    """
    imgs, ys = [], []
    for item in batch:
        if len(item) == 3:
            img, y, _txt = item
        elif len(item) == 4:
            img, y, _txt, _path = item
        else:
            raise ValueError(f"Unexpected item length {len(item)} (expected 3 or 4)")
        imgs.append(img)
        if torch.is_tensor(y):
            ys.append(int(y.item()))
        else:
            ys.append(int(y))
    return torch.stack(imgs, 0), torch.tensor(ys, dtype=torch.long)


def build_text_features(model, tokenizer, class_index_json, prompts_json, device):
    """
    Build one text embedding per class using the first prompt per class.
    """
    with open(class_index_json, "r") as f:
        class_index = json.load(f)
    with open(prompts_json, "r") as f:
        all_prompts = json.load(f)

    idx_to_label = {v: k for k, v in class_index.items()}
    num_classes = len(idx_to_label)

    texts = []
    for i in range(num_classes):
        label = idx_to_label[i]
        plist = all_prompts.get(label, [label])
        prompt = plist[0] if len(plist) > 0 else label
        texts.append(prompt)

    with torch.no_grad():
        tokenized = tokenizer(texts).to(device)
        with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_model(model, text_features, loader, device, desc="Eval"):
    """
    Compute top-1 / top-5 accuracy (unconstrained) on given loader.
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, ys in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()  # (B, C)

            # top-5 indices
            _, top5_idx = logits.topk(5, dim=1)  # (B, 5)
            # top-1
            pred1 = top5_idx[:, 0]

            top1_correct += (pred1 == ys).sum().item()
            # ys in top5?
            top5_correct += (top5_idx == ys.unsqueeze(1)).any(dim=1).sum().item()
            total += ys.size(0)

    top1 = top1_correct / total
    top5 = top5_correct / total
    return top1, top5, total


def load_finetuned_model(model_name, ckpt_path, device):
    """
    Create the CLIP model skeleton and load your fine-tuned checkpoint.
    """
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,            # start from random
        precision="fp16" if device == "cuda" else "fp32",
    )
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)

    return model


def load_base_openai_model(model_name, device):
    """
    Load the base OpenAI-pretrained CLIP model (no fine-tuning).
    """
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai",
        precision="fp16" if device == "cuda" else "fp32",
    )
    model = model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True, help="Fine-tuned checkpoint path")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset: test split only
    ds_test = CSVClipDataset(
        manifest_csv=args.manifest,
        splits_csv=args.splits,
        class_index_json=args.class_index,
        prompts_json=args.prompts,
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

    tokenizer = open_clip.get_tokenizer(args.model)

    # ---------------------------
    # 1) Base OpenAI CLIP
    # ---------------------------
    print("\n=== Evaluating base OpenAI CLIP ===")
    base_model = load_base_openai_model(args.model, device)
    base_text_features = build_text_features(
        base_model, tokenizer, args.class_index, args.prompts, device
    )
    base_top1, base_top5, total = evaluate_model(
        base_model, base_text_features, loader_test, device, desc="Eval (base)"
    )
    print(f"[Base OpenAI CLIP] top1={base_top1:.3f}, top5={base_top5:.3f}  (N={total})")

    # ---------------------------
    # 2) Fine-tuned CLIP
    # ---------------------------
    print("\n=== Evaluating fine-tuned CLIP ===")
    ft_model = load_finetuned_model(args.model, args.ckpt, device)
    ft_text_features = build_text_features(
        ft_model, tokenizer, args.class_index, args.prompts, device
    )
    ft_top1, ft_top5, total2 = evaluate_model(
        ft_model, ft_text_features, loader_test, device, desc="Eval (finetuned)"
    )
    print(f"[Fine-tuned CLIP] top1={ft_top1:.3f}, top5={ft_top5:.3f}  (N={total2})")

    # ---------------------------
    # 3) Improvement summary
    # ---------------------------
    print("\n=== Improvement ===")
    d1 = ft_top1 - base_top1
    d5 = ft_top5 - base_top5
    print(f"Δ top1 = {d1:+.3f}")
    print(f"Δ top5 = {d5:+.3f}")


if __name__ == "__main__":
    main()
