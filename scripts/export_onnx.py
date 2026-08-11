#!/usr/bin/env python3
"""Export the winning checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = args.weights
    if not weights:
        best_json = ROOT / "experiments" / "best_run.json"
        if best_json.exists():
            weights = json.loads(best_json.read_text())["winner"]["best_weights"]
        else:
            weights = str(ROOT / "weights" / "best.pt")
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path
    if not weights_path.exists():
        raise SystemExit(f"[export] Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    print(f"[export] Exporting {weights_path} → ONNX (imgsz={args.imgsz})")
    out = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    out_path = Path(str(out))
    dest = ROOT / "weights" / "best.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, dest)
    print(f"[export] ONNX saved → {dest}")


if __name__ == "__main__":
    main()
