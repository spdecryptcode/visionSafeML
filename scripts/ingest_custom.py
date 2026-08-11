#!/usr/bin/env python3
"""
Ingest Label Studio YOLO export (or a YOLO folder) into data/custom,
split 80/20, and rewrite configs/data_with_custom.yaml to mix custom
train images with the public Construction-PPE train set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CUSTOM = ROOT / "data" / "custom"
PUBLIC = ROOT / "data" / "construction-ppe"
COMBINED_YAML = ROOT / "configs" / "data_with_custom.yaml"
CUSTOM_DATA_YAML = ROOT / "configs" / "data_custom_holdout.yaml"


CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


def collect_pairs(src_images: Path, src_labels: Path) -> list[tuple[Path, Path]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for img in sorted(src_images.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        lbl = src_labels / f"{img.stem}.txt"
        if not lbl.exists():
            # try nested labels mirroring structure
            alt = src_labels / img.relative_to(src_images).with_suffix(".txt")
            if alt.exists():
                lbl = alt
            else:
                print(f"[ingest] WARNING: no label for {img.name}, skipping")
                continue
        pairs.append((img, lbl))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        default=str(CUSTOM / "raw" / "images"),
        help="Folder of labeled custom images (Label Studio YOLO export)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=str(CUSTOM / "raw" / "labels"),
        help="Folder of YOLO .txt labels",
    )
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_images = Path(args.images)
    src_labels = Path(args.labels)
    if not src_images.exists():
        raise SystemExit(
            f"[ingest] Images not found: {src_images}\n"
            "Export from Label Studio into data/custom/raw/ (see docs/LABELING_GUIDE.md)."
        )

    pairs = collect_pairs(src_images, src_labels)
    if len(pairs) < 5:
        raise SystemExit(f"[ingest] Need at least 5 labeled images, found {len(pairs)}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_hold = max(1, int(len(pairs) * args.holdout))
    holdout, train = pairs[:n_hold], pairs[n_hold:]

    for split, items in [("train", train), ("holdout", holdout)]:
        img_dir = CUSTOM / "images" / split
        lbl_dir = CUSTOM / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / f"{img.stem}.txt")

    # Build a combined dataset root with train lists via YOLO multi-dir support.
    # Ultralytics accepts a list for train.
    combined_root = ROOT / "data" / "combined"
    if combined_root.exists():
        shutil.rmtree(combined_root)
    (combined_root / "images" / "train").mkdir(parents=True)
    (combined_root / "labels" / "train").mkdir(parents=True)
    # Symlink/copy public train + custom train
    for src_root, split_name in [(PUBLIC, "train"), (CUSTOM, "train")]:
        for img in (src_root / "images" / ("train" if src_root == PUBLIC else "train")).glob("*"):
            if not img.is_file():
                continue
            dest_name = f"{src_root.name}_{img.name}"
            shutil.copy2(img, combined_root / "images" / "train" / dest_name)
            lbl = src_root / "labels" / "train" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, combined_root / "labels" / "train" / f"{Path(dest_name).stem}.txt")

    # Val/test stay public official splits (copy references via yaml path)
    # For simplicity, point val/test to public dirs using absolute path in yaml.
    combined_cfg = {
        "path": str(combined_root.resolve()),
        "train": "images/train",
        "val": str((PUBLIC / "images" / "val").resolve()),
        "test": str((PUBLIC / "images" / "test").resolve()),
        "names": CLASS_NAMES,
        "nc": 11,
    }
    # Ultralytics needs labels next to val/test — keep using public dataset path for val/test
    # Override: use public dataset as path and list both train folders.
    combined_cfg = {
        "path": str(PUBLIC.resolve()),
        "train": [
            str((PUBLIC / "images" / "train").resolve()),
            str((CUSTOM / "images" / "train").resolve()),
        ],
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": 11,
    }
    with open(COMBINED_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(combined_cfg, f, sort_keys=False)

    holdout_cfg = {
        "path": str(CUSTOM.resolve()),
        "train": "images/train",  # unused
        "val": "images/holdout",
        "test": "images/holdout",
        "names": CLASS_NAMES,
        "nc": 11,
    }
    with open(CUSTOM_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(holdout_cfg, f, sort_keys=False)

    manifest = {
        "custom_total": len(pairs),
        "custom_train": len(train),
        "custom_holdout": len(holdout),
        "combined_yaml": str(COMBINED_YAML),
        "holdout_yaml": str(CUSTOM_DATA_YAML),
    }
    (ROOT / "experiments" / "custom_ingest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print("[ingest] Next: evaluate winner on custom holdout, then `make train-custom`.")


if __name__ == "__main__":
    main()
