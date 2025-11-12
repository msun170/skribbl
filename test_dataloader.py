#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

# Import your dataset (CSVClipDataset) and keep back-compat alias if you used it
from dataset_csv import CSVClipDataset as SketchCSV

def _unpack_batch(batch):
    """
    Supports either:
      - dict batches: {'images', 'labels', 'texts', 'paths'} or similar keys
      - tuple/list batches: (images, labels, texts, paths) or (images, labels, texts)
    Returns: (images, labels, texts, paths or None)
    """
    images = labels = texts = paths = None

    if isinstance(batch, dict):
        images = batch.get('images') or batch.get('image')
        labels = batch.get('labels') or batch.get('label')
        texts  = batch.get('texts')  or batch.get('text')
        paths  = batch.get('paths')  or batch.get('path')
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            images, labels = batch[0], batch[1]
        if len(batch) >= 3:
            texts = batch[2]
        if len(batch) >= 4:
            paths = batch[3]
    return images, labels, texts, paths

def main():
    p = argparse.ArgumentParser(description="Smoke-test CSV dataset + DataLoader.")
    p.add_argument("--manifest", required=True, help="Path to manifest CSV (linux_training_manifest*.csv)")
    p.add_argument("--splits", default=None, help="Path to splits.csv (optional but supported)")
    p.add_argument("--class_index", default=None, help="class_index.json (optional)")
    p.add_argument("--prompts", default=None, help="prompts.json (optional)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which split to use")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Limit dataset to N samples (0 = no limit)")
    p.add_argument("--batches", type=int, default=2, help="How many batches to print")
    p.add_argument("--shuffle", action="store_true", help="Shuffle batches")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    if args.splits:
        assert Path(args.splits).exists(), f"Splits CSV not found: {args.splits}"
    if args.class_index:
        assert Path(args.class_index).exists(), f"class_index.json not found: {args.class_index}"
    if args.prompts:
        assert Path(args.prompts).exists(), f"prompts.json not found: {args.prompts}"

    # Build dataset
    ds = SketchCSV(
        manifest_csv=str(manifest_path),
        splits_csv=args.splits,
        class_index_json=args.class_index,
        prompts_json=args.prompts,
        split=args.split,
        img_size=args.image_size,
        # prompt_strategy = "random"  # uncomment if you expose this in your dataset class
    )

    print(f"Dataset size (total for split='{args.split}'): {len(ds)}")

    # Limit if requested
    if args.limit and args.limit > 0:
        ds = Subset(ds, range(min(args.limit, len(ds))))
        print(f"Dataset size (limited): {len(ds)}")

    # DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # Iterate a few batches
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        images, labels, texts, paths = _unpack_batch(batch)

        # Print shapes / dtypes
        if isinstance(images, torch.Tensor):
            print(f"\nBatch {bi+1}:")
            print(f"  images:  {tuple(images.shape)}  (dtype={images.dtype})")
        else:
            print(f"\nBatch {bi+1}: images is not a tensor (type={type(images)})")

        if isinstance(labels, torch.Tensor):
            print(f"  labels:  {tuple(labels.shape)}  min={labels.min().item()} max={labels.max().item()}")
        else:
            try:
                print(f"  labels:  len={len(labels)}  type={type(labels)}")
            except Exception:
                print(f"  labels:  type={type(labels)}")

        # Show a sample text/path if present
        sample_text = None
        sample_path = None
        try:
            if texts is not None:
                sample_text = texts[0] if isinstance(texts, (list, tuple)) else texts
            if paths is not None:
                sample_path = paths[0] if isinstance(paths, (list, tuple)) else paths
        except Exception:
            pass

        if sample_text is not None:
            print(f"  texts[0]: {repr(sample_text)[:120]}")
        if sample_path is not None:
            print(f"  path[0]:  {sample_path}")

if __name__ == "__main__":
    main()
