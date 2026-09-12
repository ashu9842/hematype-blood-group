# HEMATYPE/8 — Multi-modal Blood Group Estimation

> ⚠️ **Research prototype. NOT A MEDICAL DEVICE.** Predicting blood group from
> fingerprint biometrics has **no established scientific or clinical validity**.
> This project is an engineering demonstration of a dual-input deep-learning
> pipeline. Never use its output for transfusion, cross-matching, or diagnosis.
> Blood typing must be performed by an accredited laboratory using serology.

A two-stream deep network that fuses a **fingerprint biometric** and a
**microscopic blood-smear / hemagglutination image** to estimate one of the
eight ABO/Rh phenotypes (`A+ A- B+ B- AB+ AB- O+ O-`).

---

## 1. System architecture & model pipeline

**Preprocessing**
- Fingerprint: grayscale → **CLAHE** (clip 2.5, 8×8 tiles) to accentuate ridge
  structure → replicate to 3 channels → ImageNet normalization.
- Smear: RGB → non-local-means denoise → resize → ImageNet normalization.
- Augmentation (train only): affine/perspective jitter, color jitter, random
  erasing.

**Class imbalance**
- `WeightedRandomSampler` + class-weighted cross-entropy so rarer phenotypes
  (AB−, B−) are not drowned out.

**Backbone & fusion**
- Two **EfficientNet-B2** backbones (ImageNet transfer learning), one per
  modality (1408-d features each).
- **Feature-concatenation fusion** → `Linear(2816→512) → BN → SiLU → Dropout →
  Linear(512→8)`.
- Auxiliary per-stream heads enable **weighted soft-voting / late fusion** at
  inference (`0.6·fused + 0.15·fp + 0.25·smear`).

**Reaching 90–95% validation accuracy**
- Transfer learning + 3-epoch backbone freeze warmup, then discriminative
  fine-tuning.
- AdamW + **OneCycle** (cosine) LR schedule, AMP mixed precision.
- **Mixup** (α=0.2, 50% of batches), **dropout** (0.4), **label smoothing**
  (0.1), deep supervision on the auxiliary heads.
- Metrics: confusion matrix, precision/recall/F1 per class, macro-F1 for model
  selection.

> Reported accuracy targets are achievable only on the *specific benchmark
> datasets* and reflect dataset artifacts, not a biological signal.

## 2. Files
| File | Purpose |
|------|---------|
| `model.py` | Preprocessing + `DualStreamBloodGroupNet` + ensemble inference |
| `train_and_export.py` | Data loader, training loop, evaluation, weight export |
| `main.py` | FastAPI `/api/predict` + `/api/health` |
| `requirements.txt` / `Dockerfile` | Environment & container |

## 3. Datasets (reference)
- Fingerprint: Kaggle *Fingerprint Based Blood Group Dataset*, SOCOFing (modified).
- Smear/agglutination: public blood-cell / slide-agglutination sets on
  Mendeley / Kaggle.

Expected layout (paired samples share a stem/id):
```
data/{fingerprint,smear}/{train,val,test}/<CLASS>/<id>.<ext>
```

## 4. Setup & execution

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (produces model.pt)
python train_and_export.py --data ./data --epochs 60 --out model.pt

# Serve
uvicorn main:app --host 0.0.0.0 --port 8000
```

Docker:
```bash
docker build -t hematype8 .
docker run -p 8000:8000 hematype8
```

Example request:
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "fingerprint=@sample_fp.bmp" \
  -F "smear=@sample_smear.jpg"
```

## 5. Frontend
The React/Vite/Tailwind UI (repo root, `src/`) posts both images to
`/api/predict`. When no backend is reachable it falls back to a deterministic
in-browser estimator so the interface stays interactive — that fallback is a
demo aid and carries no clinical meaning. Point the frontend at this service by
proxying `/api` to `http://localhost:8000` (Vite `server.proxy`) or serving both
behind one reverse proxy.
