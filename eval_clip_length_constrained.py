#!/usr/bin/env python3
# eval_clip_length_constrained.py

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from dataset_csv import CSVClipDataset


def norm_label_for_length(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z]", "", s)
    return s


def letter_len(s: str) -> int:
    return len(norm_label_for_length(s))


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


def load_checkpoint_into_model(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
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

    idx_to_label = {v: k for k, v in class_index.items()}
    num_classes = len(idx_to_label)

    class_names = []
    texts_per_class = []

    for i in range(num_classes):
        label = idx_to_label[i]
        class_names.append(label)
        plist = all_prompts.get(label, [label])
        prompt = plist[0] if len(plist) > 0 else label
        texts_per_class.append(prompt)

    class_lens = [letter_len(c) for c in class_names]

    with torch.no_grad():
        tokenized = tokenizer(texts_per_class).to(device)
        with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features, class_names, class_lens, idx_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_metrics", default=None, help="JSON file for length-wise and overall stats")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset: test split
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
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_eval,
    )

    # Model + checkpoint
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=None)
    model = load_checkpoint_into_model(model, args.ckpt)
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.model)

    # Precompute text features for all classes + class lengths
    text_features, class_names, class_lens, idx_to_label = build_class_text_features(
        model, tokenizer, args.class_index, args.prompts, device
    )
    class_lens = torch.tensor(class_lens, device=device, dtype=torch.long)

    # Per-length stats
    per_len = {}  # L -> dict(top1_correct, top5_correct, total)

    with torch.no_grad():
        for images, ys in tqdm(loader, desc="Eval (length constrained)"):
            images = images.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()  # (B, C)

            B = ys.size(0)

            for i in range(B):
                y = ys[i].item()
                # ground-truth class name & length
                label_name = idx_to_label[y]
                L = letter_len(label_name)

                if L not in per_len:
                    per_len[L] = {"top1_correct": 0, "top5_correct": 0, "total": 0}

                per_len[L]["total"] += 1

                # mask classes by matching length
                # boolean mask over all classes
                len_mask = (class_lens == L)  # (C,)
                sample_logits = logits[i]
                masked_logits = sample_logits.masked_fill(~len_mask, -1e4)

                # if no class has this length (weird edge case), skip
                if (~torch.isfinite(masked_logits)).all():
                    continue

                topk_vals, topk_idx = masked_logits.topk(5, dim=0)  # (5,)

                pred1 = topk_idx[0].item()
                if pred1 == y:
                    per_len[L]["top1_correct"] += 1
                if int(y) in topk_idx.tolist():
                    per_len[L]["top5_correct"] += 1

    # Aggregate & print
    overall_top1 = 0
    overall_top5 = 0
    overall_total = 0

    print("\n=== Length-constrained results ===")
    for L in sorted(per_len.keys()):
        d = per_len[L]
        if d["total"] == 0:
            continue
        t1 = d["top1_correct"] / d["total"]
        t5 = d["top5_correct"] / d["total"]
        overall_top1 += d["top1_correct"]
        overall_top5 += d["top5_correct"]
        overall_total += d["total"]
        print(f"len={L:2d}  n={d['total']:5d}  top1={t1:.3f}  top5={t5:.3f}")

    if overall_total > 0:
        o1 = overall_top1 / overall_total
        o5 = overall_top5 / overall_total
        print(f"\nOVERALL (length-constrained): top1={o1:.3f}, top5={o5:.3f}")
    else:
        o1 = o5 = 0.0
        print("\nOVERALL (length-constrained): no samples?")

    if args.out_metrics:
        out = {
            "per_length": {
                str(L): {
                    "top1": float(per_len[L]["top1_correct"] / per_len[L]["total"])
                    if per_len[L]["total"] > 0 else 0.0,
                    "top5": float(per_len[L]["top5_correct"] / per_len[L]["total"])
                    if per_len[L]["total"] > 0 else 0.0,
                    "total": int(per_len[L]["total"]),
                }
                for L in per_len
            },
            "overall": {
                "top1": float(o1),
                "top5": float(o5),
                "total": int(overall_total),
            },
            "model": args.model,
            "ckpt": args.ckpt,
            "img_size": int(args.img_size),
        }
        out_path = Path(args.out_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote metrics to {out_path}")
        

if __name__ == "__main__":
    main()
