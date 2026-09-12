"""
prepare_data.py — Build the paired dataset layout expected by train_and_export.py.

Reality check
-------------
Public fingerprint datasets (SOCOFing / Kaggle blood-group fingerprint) and
public blood-smear/agglutination datasets are from DIFFERENT people. There is
no real "same subject" pairing between the two modalities. This script builds a
*class-consistent* pairing: for each blood-group class it pairs fingerprint
images with smear images of the SAME class label. That lets the dual-stream
pipeline run end-to-end, but the fused result is a modelling exercise, NOT a
biological measurement — see README disclaimer.

Input (organize your raw downloads like this first):
    raw/fingerprint/<CLASS>/*.bmp      # e.g. raw/fingerprint/A+/1.bmp
    raw/smear/<CLASS>/*.jpg            # e.g. raw/smear/A+/img_001.jpg

Output (what the trainer reads):
    data/{fingerprint,smear}/{train,val,test}/<CLASS>/<stem>.<ext>

Run:
    python prepare_data.py --raw ./raw --out ./data --val 0.15 --test 0.10
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
IMG_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT)


def split(items: list, val: float, test: float):
    n = len(items)
    n_test = int(n * test)
    n_val = int(n * val)
    return {
        "test": items[:n_test],
        "val": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }


def main(args):
    random.seed(args.seed)
    raw = Path(args.raw)
    out = Path(args.out)
    summary = []

    for cls in CLASSES:
        fp_imgs = list_images(raw / "fingerprint" / cls)
        sm_imgs = list_images(raw / "smear" / cls)
        if not fp_imgs or not sm_imgs:
            print(f"[skip] {cls}: fp={len(fp_imgs)} sm={len(sm_imgs)}")
            continue

        random.shuffle(fp_imgs)
        random.shuffle(sm_imgs)
        # pair by index; cycle the shorter list so nothing is wasted
        n = max(len(fp_imgs), len(sm_imgs))
        pairs = [
            (fp_imgs[i % len(fp_imgs)], sm_imgs[i % len(sm_imgs)])
            for i in range(n)
        ]

        for sp, group in split(pairs, args.val, args.test).items():
            for stem_i, (fp, sm) in enumerate(group):
                stem = f"{cls.replace('+','p').replace('-','m')}_{stem_i:05d}"
                for modality, src in (("fingerprint", fp), ("smear", sm)):
                    dst_dir = out / modality / sp / cls
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst_dir / f"{stem}{src.suffix.lower()}")

        summary.append((cls, len(fp_imgs), len(sm_imgs), n))

    print("\nclass    fp_src  sm_src  paired")
    for c, a, b, n in summary:
        print(f"{c:<6} {a:>7} {b:>7} {n:>7}")
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="./raw")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
