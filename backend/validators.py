

from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

FINGERPRINT_VAL_PATH = Path("fingerprint_validator.pt")
SMEAR_VAL_PATH = Path("smear_validator.pt")
VALID_THRESHOLD = 0.50

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _build_mobilenet() -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, 1)
    return m


def load_validator(path: Path, device: str) -> nn.Module | None:
    if not path.exists():
        return None
    m = _build_mobilenet()
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    m.eval()
    return m


def is_valid(model: nn.Module, image_bytes: bytes, device: str) -> bool:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = _TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).item()
    return prob >= VALID_THRESHOLD