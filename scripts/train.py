#!/usr/bin/env python3
"""Train a YOLO model from a YAML config. Logs to W&B when available."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            print("[train] MPS not available — falling back to CPU")
        except Exception:
            print("[train] torch/MPS check failed — falling back to CPU")
        return "cpu"
    return requested


def maybe_init_wandb(run_name: str, cfg: dict) -> None:
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode == "disabled":
        return
    try:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "ppe-safety-vision")
        entity = os.environ.get("WANDB_ENTITY") or None
        # Ultralytics also auto-logs; this ensures project naming is consistent.
        if wandb.run is None:
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=cfg,
                resume="allow",
            )
            print(f"[train] W&B run: {wandb.run.url if wandb.run else run_name}")
    except Exception as e:
        print(f"[train] W&B not active ({e}). Training continues with local plots only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPE detector from config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_baseline.yaml",
        help="Path to train_*.yaml",
    )
    args = parser.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)

    os.chdir(ROOT)
    data_cfg = cfg.get("data", "configs/data.yaml")
    data_path = ROOT / data_cfg if not Path(data_cfg).is_absolute() else Path(data_cfg)
    if not (ROOT / "data" / "construction-ppe" / "images" / "train").exists():
        print("[train] Dataset missing. Run: python scripts/prepare_data.py")
        sys.exit(1)

    from ultralytics import YOLO

    device = resolve_device(str(cfg.get("device", "mps")))
    run_name = str(cfg.get("name", cfg_path.stem))
    maybe_init_wandb(run_name, cfg)

    # Absolute project path avoids Ultralytics nested runs/detect/... quirks
    project_dir = ROOT / "runs" / "detect"
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg.get("model", "yolo11n.pt"))
    print(f"[train] Starting run={run_name} model={cfg.get('model')} device={device}")
    # AMP on Apple MPS can hard-kill the process (no Python traceback).
    # Prefer explicit YAML `amp:`; otherwise disable AMP on MPS.
    if "amp" in cfg:
        amp = bool(cfg["amp"])
    else:
        amp = device != "mps"
    workers = int(cfg.get("workers", 4))
    if device == "mps":
        workers = 0

    train_kwargs = dict(
        data=str(data_path),
        epochs=int(cfg.get("epochs", 100)),
        patience=int(cfg.get("patience", 20)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=device,
        seed=int(cfg.get("seed", 42)),
        lr0=float(cfg.get("lr0", 0.01)),
        optimizer=str(cfg.get("optimizer", "auto")),
        project=str(project_dir),
        name=run_name,
        exist_ok=bool(cfg.get("exist_ok", True)),
        plots=bool(cfg.get("plots", True)),
        save=bool(cfg.get("save", True)),
        val=bool(cfg.get("val", True)),
        workers=workers,
        amp=amp,
    )
    # Optional aug / schedule knobs from YAML
    for key in ("copy_paste", "close_mosaic", "mixup", "degrees", "scale", "mosaic", "fliplr"):
        if key in cfg:
            train_kwargs[key] = cfg[key]

    results = model.train(**train_kwargs)
    best = project_dir / run_name / "weights" / "best.pt"
    # Some Ultralytics versions still insert an extra task folder
    if not best.exists():
        for candidate in project_dir.rglob(f"{run_name}/weights/best.pt"):
            best = candidate
            break
    print(f"[train] Finished. Best weights expected at: {best}")
    # Persist a tiny local summary for select_best
    try:
        metrics = getattr(results, "results_dict", None) or {}
        summary_path = ROOT / "experiments" / f"{run_name}_summary.yaml"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "name": run_name,
                    "config": str(cfg_path),
                    "best_weights": str(ROOT / best),
                    "metrics": {k: float(v) for k, v in metrics.items()} if metrics else {},
                },
                f,
                sort_keys=False,
            )
        print(f"[train] Wrote {summary_path}")
    except Exception as e:
        print(f"[train] Could not write summary: {e}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
