"""Model loading and image prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.violations import summarize_compliance

ROOT = Path(__file__).resolve().parents[1]


def load_infer_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "configs" / "infer.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_path = Path(cfg["model"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    cfg["model"] = str(model_path)
    return cfg


def resolve_device(requested: str = "mps") -> str:
    """Prefer MPS on Apple Silicon; fall back to CPU."""
    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


class PPEDetector:
    def __init__(self, config_path: str | Path | None = None):
        self.cfg = load_infer_config(config_path)
        self.device = resolve_device(self.cfg.get("device", "mps"))
        model_path = Path(self.cfg["model"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train first (`make train-baseline`) or update configs/infer.yaml."
            )
        self.model = YOLO(str(model_path))
        self.conf = float(self.cfg.get("conf", 0.25))
        self.iou = float(self.cfg.get("iou", 0.45))
        self.imgsz = int(self.cfg.get("imgsz", 640))

    def predict(self, image_source: Any) -> dict[str, Any]:
        """Run detection + compliance summary on an image path, ndarray, or PIL image."""
        results = self.model.predict(
            source=image_source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": conf,
                        "bbox": xyxy,
                    }
                )

        summary = summarize_compliance(detections, conf_threshold=self.conf)
        annotated = result.plot()  # BGR numpy array
        return {
            "detections": detections,
            "summary": summary,
            "annotated_bgr": annotated,
            "model": self.cfg["model"],
            "conf": self.conf,
            "device": self.device,
        }
