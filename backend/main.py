"""
main.py — Production FastAPI inference service for HEMATYPE/8.

Endpoints
---------
GET  /api/health          liveness + model status
POST /api/predict         multipart: fingerprint + smear image files
                          -> {top, confidence, distribution[8], modality, ...}

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import (
    CLASSES,
    DualStreamBloodGroupNet,
    decode_image,
    ensemble_predict,
    preprocess_fingerprint,
    preprocess_smear,
)
from validators import (
    FINGERPRINT_VAL_PATH,
    SMEAR_VAL_PATH,
    is_valid,
    load_validator,
)

MODEL_PATH = Path("model.pt")
MAX_BYTES = 8 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/bmp", "image/webp", "image/tiff"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOW_CONFIDENCE_THRESHOLD = 0.45

app = FastAPI(title="HEMATYPE/8 Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_fp_validator = None
_sm_validator = None


@app.on_event("startup")
def _load_models() -> None:
    global _model, _fp_validator, _sm_validator

    # Main model
    net = DualStreamBloodGroupNet().to(DEVICE)
    if MODEL_PATH.exists():
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[startup] main model loaded from {MODEL_PATH}")
    else:
        print("[startup] WARNING: no model.pt — serving untrained weights")
    net.eval()
    _model = net

    # Validators
    _fp_validator = load_validator(FINGERPRINT_VAL_PATH, DEVICE)
    _sm_validator = load_validator(SMEAR_VAL_PATH, DEVICE)

    if _fp_validator:
        print("[startup] fingerprint validator loaded")
    else:
        print("[startup] WARNING: no fingerprint_validator.pt — skipping validation")

    if _sm_validator:
        print("[startup] smear validator loaded")
    else:
        print("[startup] WARNING: no smear_validator.pt — skipping validation")


async def _read_valid(f: UploadFile, field: str) -> bytes:
    if f.content_type not in ALLOWED:
        raise HTTPException(415, f"{field}: unsupported type {f.content_type}")
    raw = await f.read()
    if not raw:
        raise HTTPException(400, f"{field}: empty file")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"{field}: exceeds {MAX_BYTES} bytes")
    return raw


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "weights_present": MODEL_PATH.exists(),
        "fingerprint_validator": _fp_validator is not None,
        "smear_validator": _sm_validator is not None,
        "device": DEVICE,
        "classes": CLASSES,
    }


@app.post("/api/predict")
async def predict(
    fingerprint: UploadFile = File(...),
    smear: UploadFile = File(...),
) -> JSONResponse:
    if _model is None:
        raise HTTPException(503, "model not loaded")

    t0 = time.perf_counter()
    fp_raw, sm_raw = await asyncio.gather(
        _read_valid(fingerprint, "fingerprint"),
        _read_valid(smear, "smear"),
    )

    # CNN validation — block non-biometric images before main model
    def _validate():
        if _fp_validator and not is_valid(_fp_validator, fp_raw, DEVICE):
            raise ValueError("not_fingerprint")
        if _sm_validator and not is_valid(_sm_validator, sm_raw, DEVICE):
            raise ValueError("not_smear")

    def _prep():
        _validate()
        fp_t = preprocess_fingerprint(decode_image(fp_raw))
        sm_t = preprocess_smear(decode_image(sm_raw))
        return ensemble_predict(_model, fp_t, sm_t, DEVICE)

    try:
        result = await asyncio.to_thread(_prep)
    except ValueError as e:
        err = str(e)
        if err in ("not_fingerprint", "not_smear"):
            raise HTTPException(status_code=422, detail="low_confidence")
        raise HTTPException(422, err)

    result["latencyMs"] = round((time.perf_counter() - t0) * 1000)
    result["disclaimer"] = (
        "Research prototype. Fingerprint-based blood group inference has no "
        "established clinical validity. Not for medical use."
    )

    top_prob = result["confidence"]
    second_prob = result["distribution"][1]["probability"]
    if top_prob < LOW_CONFIDENCE_THRESHOLD or (top_prob - second_prob) < 0.15:
        raise HTTPException(status_code=422, detail="low_confidence")

    return JSONResponse(result)