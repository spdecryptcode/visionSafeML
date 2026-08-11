#!/usr/bin/env python3
"""
Sweep confidence thresholds on the validation split and write the best
operating point into configs/infer.yaml.

Uses a simple F1-style tradeoff on missing-PPE classes when available;
falls back to maximizing mean F1 from Ultralytics box metrics proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "configs" / "infer.yaml"


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
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
    )
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
        raise SystemExit(f"[tune] Weights not found: {weights_path}")

    device = resolve_device(args.device)
    model = YOLO(str(weights_path))
    data = ROOT / args.data

    rows = []
    for conf in args.thresholds:
        print(f"[tune] conf={conf:.2f}")
        metrics = model.val(
            data=str(data),
            split="val",
            conf=conf,
            device=device,
            plots=False,
            verbose=False,
        )
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        rows.append(
            {
                "conf": conf,
                "precision": p,
                "recall": r,
                "f1": f1,
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            }
        )

    best = max(rows, key=lambda r: (r["f1"], r["mAP50"]))
    out = ROOT / "experiments" / "threshold_sweep.json"
    out.write_text(json.dumps({"rows": rows, "selected": best}, indent=2), encoding="utf-8")

    with open(INFER, encoding="utf-8") as f:
        infer = yaml.safe_load(f)
    infer["conf"] = float(best["conf"])
    with open(INFER, "w", encoding="utf-8") as f:
        yaml.safe_dump(infer, f, sort_keys=False)

    print(f"[tune] Selected conf={best['conf']} (F1={best['f1']:.4f})")
    print(f"[tune] Wrote {out} and updated {INFER}")


if __name__ == "__main__":
    main()
