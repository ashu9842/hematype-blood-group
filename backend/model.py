"""
model.py — Dual-stream multi-modal network for 8-class ABO/Rh estimation.

Architecture
------------
Two independent EfficientNet-B2 backbones (ImageNet transfer learning):
  * Stream A  — fingerprint ridge image (CLAHE-enhanced grayscale -> 3ch)
  * Stream B  — microscopic blood smear / hemagglutination image (RGB)

Feature-concatenation fusion:
  [B2_A (1408) || B2_B (1408)] -> Linear(2816->512) -> Dropout -> Linear(512->8)

The network also exposes per-stream logits so the API can report a
soft-vote / late-fusion confidence per modality alongside the fused head.

NOTE: This is a research reference implementation. Fingerprint-based blood
group inference has no established clinical validity — see README.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 384  # EfficientNetV2-S native resolution
FEAT_DIM = 1280  # V2-S feature width


# --------------------------------------------------------------------------- #
# Preprocessing pipelines
# --------------------------------------------------------------------------- #
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


def preprocess_fingerprint(bgr: np.ndarray) -> torch.Tensor:
    """CLAHE contrast equalization to accentuate ridge structure."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = cv2.resize(eq, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)  # replicate to 3 channels
    return _to_tensor(rgb)


def preprocess_smear(bgr: np.ndarray) -> torch.Tensor:
    """RGB normalization + gentle denoise for smear / agglutination slides."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 5, 5, 7, 21)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return _to_tensor(rgb)


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _backbone() -> nn.Module:
    net = models.efficientnet_v2_s(weights=None)
    net.classifier = nn.Identity()  # -> (B, 1280)
    return net


class DualStreamBloodGroupNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, p_drop: float = 0.4):
        super().__init__()
        self.fp_backbone = _backbone()
        self.sm_backbone = _backbone()

        # auxiliary per-stream heads (deep supervision + late-fusion voting)
        self.fp_head = nn.Linear(FEAT_DIM, num_classes)
        self.sm_head = nn.Linear(FEAT_DIM, num_classes)

        # fused head (feature concatenation)
        self.fusion = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(FEAT_DIM * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes),
        )

    def forward(self, fp: torch.Tensor, sm: torch.Tensor) -> dict:
        f_fp = self.fp_backbone(fp)
        f_sm = self.sm_backbone(sm)
        fused = self.fusion(torch.cat([f_fp, f_sm], dim=1))
        return {
            "fused": fused,
            "fp": self.fp_head(f_fp),
            "sm": self.sm_head(f_sm),
        }


@torch.no_grad()
def ensemble_predict(
    model: DualStreamBloodGroupNet,
    fp_tensor: torch.Tensor,
    sm_tensor: torch.Tensor,
    device: str = "cpu",
    w_fused: float = 0.6,
    w_fp: float = 0.15,
    w_sm: float = 0.25,
) -> dict:
    """Weighted soft-voting across the fused head and both per-stream heads."""
    model.eval()
    fp_b = fp_tensor.unsqueeze(0).to(device)
    sm_b = sm_tensor.unsqueeze(0).to(device)
    out = model(fp_b, sm_b)

    p_fused = torch.softmax(out["fused"], dim=1)
    p_fp = torch.softmax(out["fp"], dim=1)
    p_sm = torch.softmax(out["sm"], dim=1)

    probs = (w_fused * p_fused + w_fp * p_fp + w_sm * p_sm).squeeze(0)
    probs = probs / probs.sum()

    top_idx = int(torch.argmax(probs).item())
    order = torch.argsort(probs, descending=True)
    distribution = [
        {"group": CLASSES[i], "probability": float(probs[i])} for i in order
    ]
    return {
        "top": CLASSES[top_idx],
        "confidence": float(probs[top_idx]),
        "distribution": distribution,
        "modality": {
            "fingerprint": float(p_fp.max()),
            "smear": float(p_sm.max()),
        },
    }
