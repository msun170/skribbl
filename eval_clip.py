# eval_clip.py
import argparse, json
from torch.utils.data import DataLoader
import torch, torch.nn.functional as F
import open_clip
from tqdm import tqdm
from dataset_csv import CSVClipDataset

def collate(batch):
    imgs, texts, ys = zip(*batch)
    return torch.stack(imgs, 0), list(texts), torch.tensor(ys, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--class_index", default="class_index.json")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CSVClipDataset(args.manifest, args.splits, args.class_index, args.prompts,
                        split="test", img_size=224, prompt_strategy="first")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, collate_fn=collate)

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=None, device=device)
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    tokenizer = open_clip.get_tokenizer(args.model)

    # Build text bank
    labels = sorted(list(set(ds.manifest["label"].tolist())))
    prompt_map = ds.prompts
    label_prompts = [prompt_map[l][0] if prompt_map.get(l) else f"a sketch of {l}" for l in labels]

    with torch.no_grad():
        tokens = tokenizer(label_prompts).to(device)
        text = model.encode_text(tokens)
        text = F.normalize(text, dim=-1)

        tot, top1, top5 = 0, 0, 0
        idx_map = [ds.class_index[l] for l in labels]

        for imgs, _, ys in tqdm(dl, desc="Test"):
            imgs, ys = imgs.to(device), ys.to(device)
            img = F.normalize(model.encode_image(imgs), dim=-1)
            sims = img @ text.t()
            k = sims.topk(5, dim=1).indices
            # map to global ids
            pred1 = [idx_map[i.item()] for i in k[:,0]]
            pred5 = [[idx_map[j.item()] for j in row] for row in k]
            y = ys.cpu().numpy()
            import numpy as np
            top1 += (np.array(pred1) == y).sum()
            top5 += sum(y0 in row for y0, row in zip(y, pred5))
            tot += len(y)

    print(f"Test: top1={top1/tot:.3f}, top5={top5/tot:.3f}")

if __name__ == "__main__":
    main()
