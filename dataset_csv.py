#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ------------ Path normalization helpers ------------

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:/")

def _norm_path(p: Any) -> str:
    """Normalize a possibly-Windows path to POSIX-like, collapse slashes,
    drop drive letters and file:// prefixes."""
    if not isinstance(p, str):
        p = str(p)
    s = p.strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    s = re.sub(r"^file:/+", "", s, flags=re.IGNORECASE)
    s = _WIN_DRIVE_RE.sub("", s)  # drop C:/ etc
    s = re.sub(r"/+", "/", s)
    return s


# ------------ Image transforms (CLIP-ish) ------------

def _build_transforms(image_size: int) -> T.Compose:
    # OpenAI CLIP mean/std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ------------ Dataset ------------

class CSVClipDataset(Dataset):
    """
    Expects a manifest CSV with at least:
      - path : path to image (absolute or relative)
      - label: class label (int or str)

    Optional:
      - splits CSV with columns: path, split in {'train','val','test'}
      - class_index_json mapping {label_str: int_idx}
      - prompts_json mapping {label_str: [prompt1, prompt2, ...]}

    Returns per item:
      (image_tensor, label_tensor(int64), prompt_text(str), path(str))
    """

    def __init__(
        self,
        manifest_csv: str,
        splits_csv: Optional[str] = None,
        class_index_json: Optional[str] = None,
        prompts_json: Optional[str] = None,
        split: str = "train",
        img_size: int = 224,
        prompt_strategy: str = "random",
        root_prefix: Optional[str] = "/workspace/skribbl",
    ) -> None:
        self.split = split
        self.prompt_strategy = prompt_strategy
        self.transform = _build_transforms(img_size)

        # Load manifest and normalize paths
        self.manifest = pd.read_csv(manifest_csv, low_memory=False)
        if "path" not in self.manifest.columns:
            raise ValueError("manifest CSV must contain a 'path' column")
        self.manifest["path"] = self.manifest["path"].map(_norm_path)

        # If paths are relative, optionally prepend a root prefix (but only for existence checks,
        # not written back; we keep manifest 'path' as stored).
        self.root_prefix = None
        if root_prefix:
            rp = Path(root_prefix)
            if rp.is_absolute():
                self.root_prefix = rp

        # Handle splits if provided
        if splits_csv is not None:
            splits_df = pd.read_csv(splits_csv, low_memory=False)
            if not {"path", "split"}.issubset(splits_df.columns):
                raise ValueError("splits CSV must contain 'path' and 'split' columns")
            splits_df["path"] = splits_df["path"].map(_norm_path)

            # Inner-join on normalized 'path'
            self.manifest = self.manifest.merge(
                splits_df[["path", "split"]], on="path", how="inner"
            )
            self.manifest = self.manifest[self.manifest["split"] == self.split].reset_index(drop=True)
        else:
            # No splits provided: tag everything as the chosen split
            self.manifest["split"] = self.split
            self.manifest = self.manifest.reset_index(drop=True)

        if len(self.manifest) == 0:
            # Helpful diagnostics
            raise ValueError(
                f"No rows left after applying split='{self.split}'. "
                "Check that your splits CSV 'path' values match your manifest 'path' values "
                "(relative vs absolute, Windows vs Linux, etc.)."
            )

        # Class index: if given, map string labels -> ints; else try to infer
        self.class_index: Dict[str, int]
        if class_index_json is not None:
            self.class_index = json.loads(Path(class_index_json).read_text(encoding="utf-8"))
            # Create inverse for readable label_str
            self.idx_to_label = {int(v): k for k, v in self.class_index.items()}
        else:
            # Build identity/enum map from manifest['label']
            labs = list(map(str, sorted(self.manifest["label"].astype(str).unique())))
            self.class_index = {lbl: i for i, lbl in enumerate(labs)}
            self.idx_to_label = {i: lbl for lbl, i in self.class_index.items()}

        # Prompts (optional). If missing, build simple defaults.
        if prompts_json is not None:
            self.prompts: Dict[str, List[str]] = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        else:
            self.prompts = {}

        # Ensure every label has at least one prompt
        for lbl_str in map(str, self.manifest["label"].astype(str).unique()):
            if lbl_str not in self.prompts or not self.prompts[lbl_str]:
                self.prompts[lbl_str] = [f"a contour drawing of a {lbl_str}"]

        # Cache columns to lists for speed
        self._paths = self.manifest["path"].tolist()
        self._labels_raw = self.manifest["label"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self._paths)

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        if self.root_prefix is not None:
            return (self.root_prefix / pp).resolve()
        return pp.resolve()

    def _pick_prompt(self, lbl_str: str) -> str:
        choices = self.prompts.get(lbl_str) or [f"a contour drawing of a {lbl_str}"]
        if self.prompt_strategy == "random" and len(choices) > 1:
            # small torch-free RNG to avoid global state
            import random
            return random.choice(choices)
        return choices[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        path_str = self._paths[idx]
        lbl_str = self._labels_raw[idx]

        # map to int id
        try:
            target = self.class_index[lbl_str]
        except KeyError:
            # If class_index missing the key, fall back by creating it on the fly
            target = self.class_index.setdefault(lbl_str, len(self.class_index))
            self.idx_to_label[target] = lbl_str

        full_path = self._resolve_path(path_str)
        if not full_path.is_file():
            # Raise a clear error so the calling script can log/skip if desired
            raise FileNotFoundError(f"Image not found: {full_path} (from '{path_str}')")

        with Image.open(full_path) as im:
            image = self.transform(im)

        text = self._pick_prompt(lbl_str)
        label_tensor = torch.tensor(target, dtype=torch.long)
        return image, label_tensor, text, str(full_path)


# Back-compat alias for older scripts
SketchCSV = CSVClipDataset
