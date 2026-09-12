"""
train_validators.py — Train fingerprint and blood smear binary validators.

Dataset folder structure:
  data/
    fingerprint/
      train/
        real/    ← SOCOFing fingerprint images (ya jo aapke paas hain)
        fake/    ← random images (ImageNet ya internet se)
      val/
        real/
        fake/
    smear/
      train/
        real/    ← blood smear images
        fake/    ← random images
      val/
        real/
        fake/

Usage:
    python train_validators.py --type fingerprint --epochs 10
    python train_validators.py --type smear --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, 1)
    return m


def train(type_: str, epochs: int, lr: float, batch: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(f"data/{type_}")
    train_ds = datasets.ImageFolder(data_dir / "train", TRANSFORM_TRAIN)
    val_ds = datasets.ImageFolder(data_dir / "val", TRANSFORM_VAL)

    # ImageFolder sorts alphabetically: fake=0, real=1
    print(f"Classes: {train_ds.classes}  (real should be index 1)")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=2)

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    out_path = f"{type_}_validator.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = (torch.sigmoid(model(imgs).squeeze(1)) > 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"loss={total_loss / len(train_dl):.4f} | val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"  ✓ Best model saved → {out_path}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["fingerprint", "smear"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    train(args.type, args.epochs, args.lr, args.batch)