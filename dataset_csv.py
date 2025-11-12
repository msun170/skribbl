import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _build_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP preproc
    ])

class CSVClipDataset(Dataset):
    """
    Expects:
      - training_manifest.csv with columns: path, label, source, title
      - splits.csv with columns: path, split (train/val/test)
      - class_index.json mapping label -> int
      - prompts.json mapping label -> [prompt variants]
    """
    def __init__(self, manifest_csv, splits_csv, class_index_json, prompts_json,
                 split="train", img_size=224, prompt_strategy="random"):
        self.manifest = pd.read_csv(manifest_csv)
        self.splits = pd.read_csv(splits_csv)
        self.manifest = self.manifest.merge(self.splits[["path","split"]], on="path", how="inner")
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)

        self.class_index = json.loads(Path(class_index_json).read_text(encoding="utf-8"))
        self.idx_to_label = {v:k for k,v in self.class_index.items()}

        self.prompts = json.loads(Path(prompts_json).read_text(encoding="utf-8"))
        # Ensure every label has at least one prompt (fallback to "a sketch of {label}")
        for lab in self.manifest["label"].unique():
            if lab not in self.prompts or not self.prompts[lab]:
                self.prompts[lab] = [f"a sketch of {lab}"]

        self.transform = _build_transforms(img_size)
        self.prompt_strategy = prompt_strategy

    def __len__(self):
        return len(self.manifest)

    def _pick_prompt(self, label):
        plist = self.prompts.get(label, [f"a sketch of {label}"])
        if self.prompt_strategy == "first":
            return plist[0]
        return random.choice(plist)

    def __getitem__(self, i):
        row = self.manifest.iloc[i]
        path = row["path"]
        label = row["label"]
        y = self.class_index[label]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        text = self._pick_prompt(label)
        return img, text, y
