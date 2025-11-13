#!/usr/bin/env python3
# eval_clip.py

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
    Accept items shaped like your dataset:
      (image, label, text) or (image, label, text, path)

    Returns:
      images: (B, C, H, W)
      labels: (B,) long
    """
    imgs, ys = [], []
    for item in batch:
        if len(item) == 3:
            img, y, _txt = item          # (image, label, text)
        elif len(item) == 4:
            img, y, _txt, _path = item   # (image, label, text, path)
        else:
            raise ValueError(f"Unexpected item length {len(item)} (expected 3 or 4)")

        imgs.append(img)

        # y might be a scalar tensor or int
        if torch.is_tensor(y):
            ys.append(int(y.item()))
        else:
            ys.append(int(y))

    images = torch.stack(imgs, 0)
    labels = torch.tensor(ys, dtype=torch.long)
    return images, labels


def load_checkpoint_into_model(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # handle different checkpoint formats
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


def build_class_text_features(model, tokenizer, class_index_json: str, prompts_json: str, device: str):
    # class_index: {label_str: int_id}
    with open(class_index_json, "r") as f:
        class_index = json.load(f)
    # prompts: {label_str: [prompt1, prompt2, ...]}
    with open(prompts_json, "r") as f:
        all_prompts = json.load(f)

    # id -> label mapping
    idx_to_label = {v: k for k, v in class_index.items()}
    num_classes = len(idx_to_label)

    texts_per_class = []
    for i in range(num_classes):
        label = idx_to_label[i]
        plist = all_prompts.get(label, [label])
        prompt = plist[0] if len(plist) > 0 else label
        texts_per_class.append(prompt)

    with torch.no_grad():
        tokenized = tokenizer(texts_per_class).to(device)
        with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features, idx_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--class_index", default="class_index.json")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out", default=None, help="Optional JSON file to write metrics")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the test split for evaluation
    ds = CSVClipDataset(
        manifest_csv=args.manifest,
        splits_csv=args.splits,
        class_index_json=args.class_index,
        prompts_json=args.prompts,
        split="test",
        img_size=args.img_size,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=collate_eval,
    )

    # Build model skeleton, then load our trained weights
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=None)
    model = load_checkpoint_into_model(model, args.ckpt)
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.model)

    # Precompute per-class text features
    text_features, idx_to_label = build_class_text_features(
        model, tokenizer, args.class_index, args.prompts, device
    )

    top1 = 0
    top5 = 0
    tot = 0

    with torch.no_grad():
        for images, ys in tqdm(loader, desc="Eval test"):
            images = images.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                sims = logit_scale * image_features @ text_features.t()  # (B, C)

            # top-5 classes for each image
            k = sims.topk(5, dim=1).indices  # (B, 5)
            pred1 = k[:, 0]

            # top-1 correct
            top1 += (pred1 == ys).sum().item()
            # top-5 correct if label appears anywhere in top-5 preds
            top5 += sum(int(y0 in row) for y0, row in zip(ys, k))
            tot += ys.size(0)

    top1_acc = top1 / tot
    top5_acc = top5 / tot

    print(f"Test: top1={top1_acc:.3f}, top5={top5_acc:.3f}")

    metrics = {
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "num_samples": int(tot),
        "model": args.model,
        "ckpt": str(args.ckpt),
        "img_size": int(args.img_size),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
