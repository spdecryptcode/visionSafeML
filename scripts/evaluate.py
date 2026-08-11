#!/usr/bin/env python3
"""Evaluate a checkpoint on the held-out test split (or custom split)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_device(requested: str) -> str:
    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt weights")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--name", type=str, default="test_eval")
    args = parser.parse_args()

    os_chdir = ROOT
    import os

    os.chdir(os_chdir)

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
        raise SystemExit(f"[eval] Weights not found: {weights_path}")

    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(str(weights_path))
    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)

    print(f"[eval] Evaluating {weights_path.name} on split={args.split} device={device}")
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        project="runs",
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    box = metrics.box
    per_class = {}
    names = metrics.names if hasattr(metrics, "names") else {}
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, cid in enumerate(box.ap_class_index):
            cname = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
            per_class[cname] = {
                "precision": float(box.p[i]) if box.p is not None else None,
                "recall": float(box.r[i]) if box.r is not None else None,
                "ap50": float(box.ap50[i]) if box.ap50 is not None else None,
                "ap": float(box.ap[i]) if box.ap is not None else None,
            }

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights_path),
        "split": args.split,
        "data": str(data_path),
        "device": device,
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "per_class": per_class,
    }
    out = ROOT / "experiments" / f"{args.name}_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"[eval] Wrote {out}")
    print("[eval] Fill docs/MODEL_CARD.md with these real numbers (do not invent).")


if __name__ == "__main__":
    main()
