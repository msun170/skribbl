#!/usr/bin/env python3
"""
Dump text embeddings for all Skribbl words using a fine-tuned CLIP model.

Outputs a .npz file containing:
  - "words":      array of shape (N,) with the exact skribbl words (dtype=object)
  - "embeddings": array of shape (N, D) float32, L2-normalized

You will load these in the browser and compare them against ONNX image embeddings.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import open_clip


def load_finetuned_clip(model_name: str, ckpt_path: str, device: str = "cpu"):
    """
    Load the open_clip model skeleton and then load your fine-tuned weights.

    Handles checkpoints saved as:
      - {"model": state_dict}
      - {"model_state_dict": state_dict}
      - raw state_dict
    """
    # We don't load any pretrained weights here; we immediately overwrite with your checkpoint
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,
        precision="fp32",
    )
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt
        )
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def read_skribbl_words(path: str) -> list[str]:
    """
    Read a skribbl words file: one word per line, ignoring blanks and '#' comments.
    """
    words: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            words.append(line)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skr_words",
        required=True,
        help="Path to skribbl words file (one word per line).",
    )
    ap.add_argument(
        "--model",
        default="ViT-B-32",
        help="CLIP model name for open_clip (default: ViT-B-32).",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to fine-tuned CLIP checkpoint (.pt).",
    )
    ap.add_argument(
        "--out",
        default="skribbl_text_embeddings.npz",
        help="Output .npz file.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Text batch size for encoding (default: 256).",
    )
    ap.add_argument(
        "--template",
        default="{word}",
        help=(
            "Prompt template. Use '{word}' as placeholder. "
            "Examples: '{word}', 'a sketch of a {word}', 'a drawing of a {word}'."
        ),
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load Skribbl words
    words = read_skribbl_words(args.skr_words)
    print(f"Loaded {len(words)} skribbl words from {args.skr_words}")

    # 2) Make text prompts using the template
    prompts = [args.template.format(word=w) for w in words]
    print(f"Example prompt: {prompts[0]!r}")

    # 3) Load fine-tuned CLIP model
    model = load_finetuned_clip(args.model, args.ckpt, device=device)

    # 4) Encode prompts in batches
    all_embeds = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            # Tokenize using open_clip tokenizer
            tokens = open_clip.tokenize(batch_prompts).to(device)

            text_features = model.encode_text(tokens)  # (B, D)
            # L2-normalize (as in training/eval)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_embeds.append(text_features.cpu())

            print(
                f"Encoded {i + len(batch_prompts):5d} / {len(prompts)} "
                f"({(i + len(batch_prompts)) / len(prompts) * 100:.1f}%)",
                end="\r",
                flush=True,
            )

    print()  # newline after progress
    embeds = torch.cat(all_embeds, dim=0)  # (N, D)
    embeds_np = embeds.numpy().astype("float32")

    print(f"Final embeddings shape: {embeds_np.shape}")

    # 5) Save to .npz
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(out_path, words=np.array(words, dtype=object), embeddings=embeds_np)
    print(f"Saved Skribbl text embeddings → {out_path.resolve()}")


if __name__ == "__main__":
    main()
