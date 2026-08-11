"""FastAPI inference service for PPE detection + violation summary."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import PPEDetector  # noqa: E402

detector: PPEDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    try:
        detector = PPEDetector()
        print(f"[api] Loaded model: {detector.cfg['model']} on {detector.device}")
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
        detector = None
    yield


app = FastAPI(
    title="PPE Safety Vision API",
    description="Construction-site PPE detection and compliance violations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if detector is not None else "model_missing",
        "model_loaded": detector is not None,
        "device": getattr(detector, "device", None),
        "conf": getattr(detector, "conf", None),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model and update configs/infer.yaml.",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    out = detector.predict(image)
    # Drop heavy annotated array from JSON response
    payload = {
        "filename": file.filename,
        "detections": out["detections"],
        "summary": out["summary"],
        "model": out["model"],
        "conf": out["conf"],
        "device": out["device"],
    }
    return JSONResponse(payload)


def main():
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
