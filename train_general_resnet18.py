#!/usr/bin/env python3
"""Train a ResNet‑18 model for offline signature classification.

This script trains a binary classifier to distinguish genuine from
forged signatures using a ResNet‑18 backbone. It assumes that
training and validation data are organised in the following
structure:

```
datasets/CEDAR/
├── train/
│   ├── genuine/
│   └── forged/
└── val/
    ├── genuine/
    └── forged/
```

The script uses transfer learning: it loads a ResNet‑18 pretrained on
ImageNet, replaces the final fully connected layer with a single
output node (for binary classification), and fine‑tunes the network
on the signature dataset. You can override hyperparameters such as
learning rate, batch size and number of epochs via command line.

Usage:
    python train_general_resnet18.py --data_dir datasets/CEDAR \
        --epochs 10 --batch_size 16 --lr 0.0001 

This will save the trained weights to ``models/resnet18_general.pt``.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


def get_data_loaders(data_dir: Path, batch_size: int) -> tuple:
    """Create training and validation data loaders.

    Args:
        data_dir: Path to the dataset directory with ``train`` and
            ``val`` subdirectories.
        batch_size: Batch size for loaders.

    Returns:
        A tuple of ``(train_loader, val_loader)``.
    """
    # Data augmentation and normalisation for training
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Just resizing and normalising for validation
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion, optimizer, epochs: int, device: torch.device) -> nn.Module:
    best_acc = 0.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    if best_model is not None:
        model.load_state_dict(best_model)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on a signature dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory containing train/val splits.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--output", type=str, default="models/resnet18_general.pt",
                        help="Where to save the trained model.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(data_dir, args.batch_size)

    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device)
    # Save model
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
