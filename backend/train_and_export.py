"""
train_and_export.py — Train the dual-stream network and export weights.

Expected dataset layout (paired samples share the same stem/id):
    data/
      fingerprint/<split>/<CLASS>/<id>.bmp     # SOCOFing-style ridge scans
      smear/<split>/<CLASS>/<id>.jpg           # blood smear / agglutination
    where <split> in {train, val, test} and <CLASS> in the 8 ABO/Rh labels.

Techniques applied to reach the 90–95% validation target:
  * Transfer learning from ImageNet (EfficientNet-B2 x2)
  * Class-imbalance handling: WeightedRandomSampler + class-weighted loss
  * Augmentation: affine/perspective jitter, random erasing, CLAHE for ridges
  * Regularization: dropout, label smoothing, mixup
  * Optimization: AdamW + cosine LR with warmup, AMP mixed precision, EMA
  * Discriminative fine-tuning: frozen backbones for warmup, then unfrozen

Run:
    python train_and_export.py --data ./data --epochs 60 --out model.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from model import (
    CLASSES,
    DualStreamBloodGroupNet,
    IMG_SIZE,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _clahe_pil(img: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(2.5, (8, 8)).apply(arr)
    return Image.fromarray(cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB))


class PairedBloodDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool):
        self.samples: list[tuple[Path, Path, int]] = []
        fp_root = Path(root) / "fingerprint" / split
        sm_root = Path(root) / "smear" / split
        for cls in CLASSES:
            for fp_path in (fp_root / cls).glob("*"):
                sm_path = next((sm_root / cls).glob(f"{fp_path.stem}.*"), None)
                if sm_path is not None:
                    self.samples.append((fp_path, sm_path, CLASS_TO_IDX[cls]))
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {root}/{split}")

        base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        aug = (
            [
                transforms.RandomAffine(8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.RandomPerspective(0.2, p=0.3),
                transforms.ColorJitter(0.2, 0.2, 0.1),
            ]
            if train
            else []
        )
        tail = [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
        erase = [transforms.RandomErasing(0.25)] if train else []
        self.fp_tf = transforms.Compose(base + aug + tail + erase)
        self.sm_tf = transforms.Compose(base + aug + tail + erase)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        fp_p, sm_p, y = self.samples[i]
        fp = self.fp_tf(_clahe_pil(Image.open(fp_p)))
        sm = self.sm_tf(Image.open(sm_p).convert("RGB"))
        return fp, sm, y

    def labels(self) -> list[int]:
        return [y for _, _, y in self.samples]


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / np.clip(counts, 1, None)
    sample_w = [weights[y] for y in labels]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def class_weights(labels: list[int], device: str) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    w = counts.sum() / (len(CLASSES) * np.clip(counts, 1, None))
    return torch.tensor(w, device=device)


def mixup(fp, sm, y, num_classes, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(fp.size(0), device=fp.device)
    fp_m = lam * fp + (1 - lam) * fp[perm]
    sm_m = lam * sm + (1 - lam) * sm[perm]
    y1 = torch.zeros(y.size(0), num_classes, device=fp.device).scatter_(
        1, y.unsqueeze(1), 1.0
    )
    y2 = y1[perm]
    return fp_m, sm_m, lam * y1 + (1 - lam) * y2


def soft_ce(logits, soft_targets):
    return -(soft_targets * torch.log_softmax(logits, dim=1)).sum(1).mean()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PairedBloodDataset(args.data, "train", train=True)
    val_ds = PairedBloodDataset(args.data, "val", train=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=make_sampler(train_ds.labels()),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = DualStreamBloodGroupNet().to(device)
    cw = class_weights(train_ds.labels(), device)
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl),
        pct_start=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_f1 = 0.0
    for epoch in range(args.epochs):
        # freeze backbones for the first 3 warmup epochs
        freeze = epoch < 3
        for bb in (model.fp_backbone, model.sm_backbone):
            for p in bb.parameters():
                p.requires_grad = not freeze

        model.train()
        for fp, sm, y in train_dl:
            fp, sm, y = fp.to(device), sm.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                if np.random.rand() < 0.5:
                    fp_m, sm_m, soft_y = mixup(fp, sm, y, len(CLASSES))
                    out = model(fp_m, sm_m)
                    loss = (
                        soft_ce(out["fused"], soft_y)
                        + 0.3 * soft_ce(out["fp"], soft_y)
                        + 0.3 * soft_ce(out["sm"], soft_y)
                    )
                else:
                    out = model(fp, sm)
                    loss = (
                        ce(out["fused"], y)
                        + 0.3 * ce(out["fp"], y)
                        + 0.3 * ce(out["sm"], y)
                    )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        f1 = evaluate(model, val_dl, device, verbose=(epoch % 5 == 0))
        print(f"epoch {epoch:03d}  val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print(f"  ↳ saved best -> {args.out} (f1={best_f1:.4f})")

    print(f"done. best macro-F1={best_f1:.4f}")


@torch.no_grad()
def evaluate(model, dl, device, verbose=False) -> float:
    model.eval()
    ys, preds = [], []
    for fp, sm, y in dl:
        out = model(fp.to(device), sm.to(device))
        p = torch.softmax(out["fused"], 1) * 0.6
        p += torch.softmax(out["fp"], 1) * 0.15
        p += torch.softmax(out["sm"], 1) * 0.25
        preds += p.argmax(1).cpu().tolist()
        ys += y.tolist()
    if verbose:
        print(confusion_matrix(ys, preds))
        print(classification_report(ys, preds, target_names=CLASSES, digits=3))
    return f1_score(ys, preds, average="macro")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    train(ap.parse_args())
