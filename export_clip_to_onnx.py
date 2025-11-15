#!/usr/bin/env python3
"""
Export the fine-tuned CLIP *image encoder* to ONNX.

Input  : (B, 3, H, W) float32, normalized like in training (same transforms)
Output : (B, D) float32 L2-normalized image embeddings

You will later:
  - Run this ONNX model via WebGPU in the extension
  - Compare its embeddings against precomputed Skribbl word embeddings
"""

import argparse
from pathlib import Path

import torch
import open_clip


def load_finetuned_model(model_name: str, ckpt_path: str, device: str = "cpu"):
    """
    Load the open_clip model skeleton and then load your fine-tuned weights.

    Handles checkpoints saved as:
      - {"model": state_dict}
      - {"model_state_dict": state_dict}
      - raw state_dict
    """
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,          # we will load YOUR weights
        precision="fp32",         # export in float32 for safety
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
    model.eval()
    return model


class ImageEncoderWrapper(torch.nn.Module):
    """
    Thin wrapper so ONNX export sees a simple forward(x) -> embeddings.
    """

    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), already normalized by your preprocessing
        feats = self.clip_model.encode_image(x)
        # L2-normalize like in training/eval
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True, help="Path to fine-tuned checkpoint (.pt)")
    ap.add_argument("--img_size", type=int, default=224, help="Input image size (H=W)")
    ap.add_argument("--out", default="clip_image_encoder.onnx", help="Output ONNX filename")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting on device: {device}")

    # 1) Load model + weights
    model = load_finetuned_model(args.model, args.ckpt, device=device)

    # 2) Wrap the image encoder
    wrapped = ImageEncoderWrapper(model).to(device)

    # 3) Dummy input: batch size 1, 3xHxW
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device, dtype=torch.float32)

    # 4) Export to ONNX
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {out_path} ...")
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print("Done!")
    print(f"Saved ONNX model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
