import argparse, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from dataset_csv import CSVClipDataset

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

def collate(batch):
    imgs, texts, ys = zip(*batch)
    return torch.stack(imgs, 0), list(texts), torch.tensor(ys, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest.csv")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--class_index", default="class_index.json")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--model", default="ViT-B-32")          # good start
    ap.add_argument("--pretrained", default="openai")       # openai weights
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)  # Ada 6000 48GB can handle 256x224 mixed-precision typically
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = CSVClipDataset(args.manifest, args.splits, args.class_index, args.prompts,
                              split="train", img_size=args.img_size, prompt_strategy="random")
    val_ds   = CSVClipDataset(args.manifest, args.splits, args.class_index, args.prompts,
                              split="val", img_size=args.img_size, prompt_strategy="first")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate)

    # Model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    model = model.to(device)
    model.train()

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    global_step = 0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def encode_texts(texts):
        # Tokenize + encode with CLIP text tower
        toks = tokenizer(texts).to(device)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            return model.encode_text(toks)

    def encode_images(imgs):
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            return model.encode_image(imgs)

    # Training loop (CLIP-style contrastive loss)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, texts, ys in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                img_feats = encode_images(imgs)
                txt_feats = encode_texts(texts)

                img_feats = F.normalize(img_feats, dim=-1)
                txt_feats = F.normalize(txt_feats, dim=-1)

                # Logit scale learned by CLIP
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * img_feats @ txt_feats.t()
                logits_per_text  = logits_per_image.t()

                # Construct contrastive targets: each position matches to itself in the batch
                targets = torch.arange(len(imgs), device=imgs.device)
                loss_i = F.cross_entropy(logits_per_image, targets)
                loss_t = F.cross_entropy(logits_per_text, targets)
                loss = (loss_i + loss_t) / 2

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).step(optimizer)
            scaler.update()

            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / max(1, len(train_loader))
        torch.save({"epoch": epoch+1, "model": model.state_dict()}, Path(args.out_dir)/f"clip_{args.model}_e{epoch+1}.pt")
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # quick eval @ val split (top1/top5 over label names)
        top1, top5 = evaluate(model, tokenizer, val_loader, device)
        print(f"Val: top1={top1:.3f}, top5={top5:.3f}")

def evaluate(model, tokenizer, loader, device):
    model.eval()
    import numpy as np
    # Build text bank: one prompt per label (use first prompt per label)
    labels = sorted(list(set(loader.dataset.manifest["label"].tolist())))
    prompt_map = loader.dataset.prompts
    label_prompts = [prompt_map[l][0] if prompt_map.get(l) else f"a sketch of {l}" for l in labels]

    with torch.no_grad():
        text_tokens = tokenizer(label_prompts).to(device)
        text_embeds = model.encode_text(text_tokens)
        text_embeds = F.normalize(text_embeds, dim=-1)

        tot, correct1, correct5 = 0, 0, 0
        for imgs, _, ys in tqdm(loader, desc="Eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device)

            img_embeds = model.encode_image(imgs)
            img_embeds = F.normalize(img_embeds, dim=-1)

            sims = img_embeds @ text_embeds.t()
            top5_idx = sims.topk(5, dim=1).indices  # [B,5]
            # map idx->label id via labels list
            # build a map label->class_index id
            label_to_idx = loader.dataset.class_index
            idx_to_class_idx = [label_to_idx[l] for l in labels]  # aligns with text bank

            pred1 = top5_idx[:, 0].cpu().numpy()
            pred5 = top5_idx.cpu().numpy()
            y_np = ys.cpu().numpy()

            # convert text-bank index to global class id
            pred1_ids = [idx_to_class_idx[i] for i in pred1]
            pred5_ids = [[idx_to_class_idx[j] for j in row] for row in pred5]

            correct1 += (np.array(pred1_ids) == y_np).sum()
            correct5 += sum(y in row for y, row in zip(y_np, pred5_ids))
            tot += len(y_np)

    return correct1 / tot, correct5 / tot

if __name__ == "__main__":
    main()
