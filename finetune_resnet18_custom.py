# -*- coding: utf-8 -*-
"""
Fine-tune ResNet-18 on custom signature dataset with useful training knobs:
- freeze/unfreeze backbone
- weight decay
- learning-rate schedulers (cosine/step/plateau)
- save best checkpoint (by val_acc)
- CSV logging

Usage example:
python finetune_resnet18_custom.py ^
  --data_dir datasets/custom_signature_dataset_pre ^
  --weights models/resnet18_general.pth ^
  --epochs 12 ^
  --batch_size 32 ^
  --lr 1e-4 ^
  --weight_decay 1e-4 ^
  --scheduler cosine ^
  --freeze_backbone ^
  --output models/resnet18_finetuned.pth
"""
import argparse, os, csv, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np

def set_seed(seed=7):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_loaders(data_dir, batch_size):
    # بياناتك مسبقًا preprocessed؛ ما نستخدم Augmentations في val/test
    norm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=norm)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=norm)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds.class_to_idx

def build_model(weights_path=None, freeze_backbone=False, num_classes=2, device="cpu"):
    model = models.resnet18(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)

    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        # يدعم حالة حفظ كاملة أو state_dict فقط
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        #  احذف وزن/انحياز fc من الـ checkpoint لأن شكله مختلف
        drop_keys = [k for k in state.keys() if k.startswith("fc.")]
        for k in drop_keys:
            state.pop(k, None)

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded (backbone only). missing:", missing, "unexpected:", unexpected)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    model.to(device)
    return model


def make_scheduler(name, optimizer, epochs, step_size=5, gamma=0.5):
    name = (name or "none").lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    else:
        return None

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0; correct = 0; running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    val_loss = running_loss / max(1,total)
    val_acc = correct / max(1,total)
    return val_loss, val_acc

def train(args):
    set_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, class_to_idx = get_loaders(args.data_dir, args.batch_size)
    print("class_to_idx:", class_to_idx)  # should be {'forged':0,'genuine':1}

    model = build_model(args.weights, args.freeze_backbone, num_classes=2, device=device)

    # فقط باراميترات القابلة للتدريب
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs, args.step_size, args.gamma)

    criterion = nn.CrossEntropyLoss()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log.csv")

    # CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    best_val = -1.0
    since = time.time()

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0; epoch_correct = 0; epoch_total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            epoch_correct += (pred == y).sum().item()
            epoch_total += x.size(0)

        train_loss = epoch_loss / max(1,epoch_total)
        train_acc = epoch_correct / max(1,epoch_total)

        val_loss, val_acc = evaluate(model, val_loader, device)

        # schedulers
        if args.scheduler == "plateau":
            scheduler.step(val_acc) if scheduler else None
        else:
            scheduler.step() if scheduler else None

        # log
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={lr_now:.6f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{lr_now:.6f}"])

        # save best by val_acc
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), str(out_path))
            print("Saved BEST to", str(out_path), "val_acc=", f"{best_val:.4f}")

    elapsed = time.time() - since
    print("Training complete. Best val_acc:", f"{best_val:.4f}", "time_sec=", int(elapsed))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="root containing train/ and val/")
    p.add_argument("--weights", default=None, help="path to pretrained .pth (optional)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["none","cosine","step","plateau"], default="none")
    p.add_argument("--step_size", type=int, default=5, help="for step scheduler")
    p.add_argument("--gamma", type=float, default=0.5, help="for step scheduler")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--output", default="models/resnet18_finetuned.pth")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
