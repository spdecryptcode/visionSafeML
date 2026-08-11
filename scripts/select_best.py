#!/usr/bin/env python3
"""Select the best run by validation mAP50-95 from local Ultralytics results.csv files."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "detect"
OUT = ROOT / "experiments" / "best_run.json"
INFER = ROOT / "configs" / "infer.yaml"


def read_last_metrics(results_csv: Path) -> dict:
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    # Normalize keys (Ultralytics adds spaces)
    cleaned = {k.strip(): v.strip() for k, v in last.items()}
    return cleaned


def score_row(row: dict) -> tuple[float, float]:
    def get(*keys: str) -> float:
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    continue
        return -1.0

    map5095 = get("metrics/mAP50-95(B)", "mAP50-95(B)", "metrics/mAP50-95")
    map50 = get("metrics/mAP50(B)", "mAP50(B)", "metrics/mAP50")
    return map5095, map50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="*",
        default=["baseline", "more_epochs", "lr_low", "model_s"],
        help="Run directory names under runs/detect/",
    )
    args = parser.parse_args()

    candidates = []
    for name in args.runs:
        run_dir = RUNS / name
        if not (run_dir / "weights" / "best.pt").exists():
            # Fallback: search under runs/detect/**/name
            found = None
            for candidate in RUNS.rglob(f"{name}/weights/best.pt"):
                found = candidate.parent.parent
                break
            if found is None:
                print(f"[select] Skipping {name} (missing results.csv or best.pt)")
                continue
            run_dir = found
        results_csv = run_dir / "results.csv"
        best_pt = run_dir / "weights" / "best.pt"
        if not results_csv.exists() or not best_pt.exists():
            print(f"[select] Skipping {name} (missing results.csv or best.pt)")
            continue
        row = read_last_metrics(results_csv)
        map5095, map50 = score_row(row)
        candidates.append(
            {
                "name": name,
                "map50_95": map5095,
                "map50": map50,
                "best_weights": str(best_pt.resolve()),
                "results_csv": str(results_csv.resolve()),
                "run_dir": str(run_dir.resolve()),
            }
        )
        print(f"[select] {name}: mAP50-95={map5095:.4f} mAP50={map50:.4f}")

    if not candidates:
        raise SystemExit("[select] No completed runs found. Train first.")

    candidates.sort(key=lambda c: (c["map50_95"], c["map50"]), reverse=True)
    winner = candidates[0]
    payload = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "val mAP50-95",
        "winner": winner,
        "candidates": candidates,
        "note": "Winner chosen on validation metrics only. Test eval is separate.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[select] Winner: {winner['name']} → {OUT}")

    # Point infer.yaml at winner
    with open(INFER, encoding="utf-8") as f:
        infer = yaml.safe_load(f)
    # Store path relative to repo root when possible
    try:
        rel = Path(winner["best_weights"]).relative_to(ROOT)
        infer["model"] = str(rel)
    except ValueError:
        infer["model"] = winner["best_weights"]
    with open(INFER, "w", encoding="utf-8") as f:
        yaml.safe_dump(infer, f, sort_keys=False)

    # Also copy to a stable location
    stable = ROOT / "weights" / "best.pt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["best_weights"], stable)
    print(f"[select] Copied winner weights → {stable}")
    print(f"[select] Updated {INFER}")


if __name__ == "__main__":
    main()
