# test_dataloader.py
import argparse, json, os, random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset_csv.py import SketchCSV  # adjust import if needed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class_index", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--shuffle_first", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.shuffle_first:
        df = df.sample(frac=1, random_state=7)
    if args.limit > 0:
        df = df.head(args.limit)
    df.to_csv("_tmp_subset_manifest.csv", index=False)

    with open(args.class_index, "r") as f:
        class_index = json.load(f)
    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    ds = SketchCSV(
        manifest_csv="_tmp_subset_manifest.csv",
        class_index=class_index,
        prompts=prompts,
        image_size=args.image_size,
        data_root=args.data_root,
        augment=False,   # set True if you wired augs
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f"Dataset size (tested subset): {len(ds)}  |  classes: {len(class_index)}")
    for i, batch in enumerate(dl):
        imgs = batch["image"]          # [B,3,H,W]
        labels = batch["label"]        # [B]
        texts = batch["text"]          # list[str]
        paths = batch["path"]
        names = batch["label_str"]

        print(f"\nBatch {i+1}:")
        print("  images: ", imgs.shape, " (dtype=", imgs.dtype, ")")
        print("  labels: ", labels.shape, " min=", labels.min().item(), " max=", labels.max().item())
        print("  texts[0]:", texts[0])
        print("  label_str[0]:", names[0])
        print("  path[0]:", paths[0])

        if i+1 >= args.batches:
            break

    # class counts
    counts = pd.Series([x for x in df['label']]).value_counts().head(10)
    print("\n— Subset label counts (top 10) —")
    for k, v in counts.items():
        print(f"  {k:25s} {v}")

if __name__ == "__main__":
    main()
