#!/usr/bin/env python3
"""
Senior-DS style dataset optimization on the public Construction-PPE set
(no new custom photos required).

Does:
1) Empty / unreadable label cleanup report
2) Rare-class image oversampling for train (fixes no_boots starvation)
3) Writes configs/data_optimized.yaml + experiments/data_optimize_report.json

Val/test are never altered (no leakage, honest metrics).
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "construction-ppe"
OUT = ROOT / "data" / "construction-ppe-optimized"
REPORT = ROOT / "experiments" / "data_optimize_report.json"
DATA_YAML = ROOT / "configs" / "data_optimized.yaml"

# Official class ids
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

# Oversample images that contain these (violation / rare) classes
OVERSAMPLE_CLASSES = {10}  # no_boots
# Optional secondary boost for other sparse violation classes
SECONDARY_OVERSAMPLE = {8}  # no_goggle
TARGET_MIN_FRACTION = 0.05  # aim ≥5% of train boxes for primary rare class
MAX_EXTRA_COPIES = 6


def _list_images(split_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in split_dir.iterdir() if p.suffix.lower() in exts)


def _read_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            ids.append(int(float(parts[0])))
    return ids


def _box_count(label_path: Path) -> Counter:
    return Counter(_read_classes(label_path))


def analyze_train(train_imgs: Path, train_lbls: Path) -> dict:
    empty_labels = []
    missing_labels = []
    per_image_classes: dict[str, set[int]] = {}
    box_counts: Counter = Counter()

    for img in _list_images(train_imgs):
        lp = train_lbls / f"{img.stem}.txt"
        if not lp.exists():
            missing_labels.append(img.name)
            continue
        classes = _read_classes(lp)
        if not classes:
            empty_labels.append(img.name)
            continue
        per_image_classes[img.name] = set(classes)
        box_counts.update(classes)

    return {
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
        "per_image_classes": per_image_classes,
        "box_counts": box_counts,
    }


def build_oversample_plan(analysis: dict) -> list[tuple[str, int]]:
    """Return list of (image_name, extra_copies)."""
    box_counts: Counter = analysis["box_counts"]
    total = sum(box_counts.values()) or 1
    per_image: dict[str, set[int]] = analysis["per_image_classes"]

    plan: dict[str, int] = defaultdict(int)

    def boost(class_ids: set[int], target_frac: float, max_copies: int) -> None:
        for cid in class_ids:
            current = box_counts.get(cid, 0)
            target = int(total * target_frac)
            if current >= target:
                continue
            carriers = [name for name, cls in per_image.items() if cid in cls]
            if not carriers:
                continue
            # Approximate: each carrier copy adds ~current/len(carriers) boxes
            avg = max(current / len(carriers), 1)
            need = target - current
            copies_needed = int(need / avg) + 1
            # Distribute copies round-robin across carriers
            for i in range(min(copies_needed * len(carriers), max_copies * len(carriers))):
                name = carriers[i % len(carriers)]
                if plan[name] < max_copies:
                    plan[name] += 1
                    if sum(plan[n] * avg for n in plan) >= need:
                        # stop early once roughly enough
                        pass

    boost(OVERSAMPLE_CLASSES, TARGET_MIN_FRACTION, MAX_EXTRA_COPIES)
    boost(SECONDARY_OVERSAMPLE, 0.04, 2)

    return sorted(plan.items(), key=lambda x: (-x[1], x[0]))


def materialize(plan: list[tuple[str, int]], analysis: dict) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    # Copy val/test unchanged (symlinks for speed)
    for split in ("val", "test"):
        for img in _list_images(SRC / "images" / split):
            dest = OUT / "images" / split / img.name
            dest.symlink_to(img.resolve())
            lp = SRC / "labels" / split / f"{img.stem}.txt"
            if lp.exists():
                (OUT / "labels" / split / lp.name).symlink_to(lp.resolve())

    # Train: all originals + oversampled copies (physical copies for rare)
    train_imgs = SRC / "images" / "train"
    train_lbls = SRC / "labels" / "train"
    skipped = set(analysis["empty_labels"]) | set(analysis["missing_labels"])

    copied = 0
    for img in _list_images(train_imgs):
        if img.name in skipped:
            continue
        shutil.copy2(img, OUT / "images" / "train" / img.name)
        lp = train_lbls / f"{img.stem}.txt"
        shutil.copy2(lp, OUT / "labels" / "train" / lp.name)
        copied += 1

    extra = 0
    rng = random.Random(42)
    plan_list = list(plan)
    rng.shuffle(plan_list)
    for name, n_extra in plan_list:
        if name in skipped:
            continue
        src_img = train_imgs / name
        src_lbl = train_lbls / f"{Path(name).stem}.txt"
        if not src_img.exists() or not src_lbl.exists():
            continue
        stem = Path(name).stem
        suffix = Path(name).suffix
        for i in range(n_extra):
            new_name = f"{stem}__aug{i+1}{suffix}"
            shutil.copy2(src_img, OUT / "images" / "train" / new_name)
            shutil.copy2(src_lbl, OUT / "labels" / "train" / f"{stem}__aug{i+1}.txt")
            extra += 1

    # Recount train boxes
    new_counts: Counter = Counter()
    for lp in (OUT / "labels" / "train").glob("*.txt"):
        new_counts.update(_box_count(lp))

    return {
        "train_images_base": copied,
        "train_images_extra": extra,
        "train_images_total": copied + extra,
        "skipped_empty_or_missing": sorted(skipped),
        "train_box_counts_before": {
            CLASS_NAMES[k]: int(v) for k, v in sorted(analysis["box_counts"].items())
        },
        "train_box_counts_after": {
            CLASS_NAMES.get(k, str(k)): int(v) for k, v in sorted(new_counts.items())
        },
    }


def write_yaml() -> None:
    cfg = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": 11,
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    if not (SRC / "images" / "train").exists():
        raise SystemExit("Run make prepare first.")

    analysis = analyze_train(SRC / "images" / "train", SRC / "labels" / "train")
    plan = build_oversample_plan(analysis)
    stats = materialize(plan, analysis)
    write_yaml()

    report = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC),
        "output": str(OUT),
        "data_yaml": str(DATA_YAML),
        "strategy": {
            "primary_oversample_classes": [CLASS_NAMES[c] for c in OVERSAMPLE_CLASSES],
            "secondary_oversample_classes": [CLASS_NAMES[c] for c in SECONDARY_OVERSAMPLE],
            "target_min_fraction_primary": TARGET_MIN_FRACTION,
            "max_extra_copies_per_image": MAX_EXTRA_COPIES,
            "val_test_untouched": True,
        },
        "empty_labels": analysis["empty_labels"],
        "missing_labels": analysis["missing_labels"],
        "oversample_plan_top": plan[:30],
        **stats,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "train_images_total": stats["train_images_total"],
        "extra_copies": stats["train_images_extra"],
        "skipped": len(stats["skipped_empty_or_missing"]),
        "no_boots_before": stats["train_box_counts_before"].get("no_boots"),
        "no_boots_after": stats["train_box_counts_after"].get("no_boots"),
        "yaml": str(DATA_YAML),
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
