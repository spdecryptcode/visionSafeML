#!/usr/bin/env python3
"""Download Construction-PPE, audit splits/labels, write data manifest + data card stats."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import yaml

try:
    import cv2
    import imagehash
    from PIL import Image
except ImportError as e:
    print(f"Missing dependency: {e}. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "construction-ppe"
CONFIG = ROOT / "configs" / "data.yaml"
MANIFEST = ROOT / "experiments" / "data_manifest.json"
AUDIT_DIR = ROOT / "experiments" / "audit_samples"
DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
)


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _normalize_layout() -> None:
    """Zip extracts either into data/construction-ppe/ or flat into data/images|labels."""
    marker = DATA_DIR / "images" / "train"
    if marker.exists() and any(marker.iterdir()):
        return

    # Flat extract: data/images, data/labels
    flat_images = DATA_DIR.parent / "images"
    flat_labels = DATA_DIR.parent / "labels"
    if flat_images.exists() and (flat_images / "train").exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("images", "labels", "LICENSE", "data.yaml"):
            src = DATA_DIR.parent / name
            dest = DATA_DIR / name
            if src.exists() and not dest.exists():
                shutil.move(str(src), str(dest))
        return

    # Nested extract under another folder name
    for c in DATA_DIR.parent.iterdir():
        if not c.is_dir() or c.name in {"custom", "construction-ppe"}:
            continue
        if (c / "images" / "train").exists():
            if c != DATA_DIR:
                if DATA_DIR.exists():
                    shutil.rmtree(DATA_DIR)
                shutil.move(str(c), str(DATA_DIR))
            return


def download_and_extract() -> None:
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / "images" / "train"
    if marker.exists() and any(marker.iterdir()):
        print(f"[prepare] Dataset already present at {DATA_DIR}")
        return

    zip_path = ROOT / "data" / "construction-ppe.zip"
    print(f"[prepare] Downloading Construction-PPE (~178MB)…")
    urlretrieve(DOWNLOAD_URL, zip_path)
    print(f"[prepare] Extracting to {DATA_DIR.parent}…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR.parent)

    _normalize_layout()

    if zip_path.exists():
        zip_path.unlink()

    marker = DATA_DIR / "images" / "train"
    if not marker.exists() or not any(marker.iterdir()):
        raise RuntimeError(
            f"Dataset extract failed — expected images under {DATA_DIR}/images/train"
        )
    print(f"[prepare] Dataset ready at {DATA_DIR}")


def rewrite_data_yaml() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(DATA_DIR.resolve())
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def list_images(split: str) -> list[Path]:
    img_dir = DATA_DIR / "images" / split
    if not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])


def label_path_for(image: Path, split: str) -> Path:
    # YOLO layout: labels/<split>/<stem>.txt
    return DATA_DIR / "labels" / split / f"{image.stem}.txt"


def count_classes(split: str, names: dict) -> Counter:
    counts: Counter = Counter()
    for img in list_images(split):
        lp = label_path_for(img, split)
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid = int(line.split()[0])
            counts[names.get(cid, str(cid))] += 1
    return counts


def check_filename_overlap(splits: list[str]) -> dict:
    stems = {s: {p.stem for p in list_images(s)} for s in splits}
    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            shared = sorted(stems[a] & stems[b])
            if shared:
                overlaps[f"{a}_vs_{b}"] = shared[:20]
    return overlaps


def perceptual_hash_collisions(splits: list[str], sample_per_split: int = 80) -> list[dict]:
    """Cheap near-duplicate check across splits using average hash."""
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rng = random.Random(42)
    for split in splits:
        imgs = list_images(split)
        sample = imgs if len(imgs) <= sample_per_split else rng.sample(imgs, sample_per_split)
        for p in sample:
            try:
                ph = str(imagehash.average_hash(Image.open(p)))
                buckets[ph].append((split, p.name))
            except Exception:
                continue
    collisions = []
    for ph, items in buckets.items():
        split_set = {s for s, _ in items}
        if len(split_set) > 1:
            collisions.append({"hash": ph, "items": items})
    return collisions[:25]


def image_stats(split: str, limit: int = 200) -> dict:
    imgs = list_images(split)
    rng = random.Random(42)
    sample = imgs if len(imgs) <= limit else rng.sample(imgs, limit)
    widths, heights, aspects = [], [], []
    for p in sample:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        widths.append(w)
        heights.append(h)
        aspects.append(round(w / max(h, 1), 3))
    if not widths:
        return {}
    return {
        "sampled": len(widths),
        "width_min": min(widths),
        "width_max": max(widths),
        "width_mean": round(sum(widths) / len(widths), 1),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_mean": round(sum(heights) / len(heights), 1),
        "aspect_mean": round(sum(aspects) / len(aspects), 3),
    }


def save_audit_mosaic(split: str, names: dict, n: int = 16) -> Path | None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = list_images(split)
    if not imgs:
        return None
    rng = random.Random(42)
    sample = rng.sample(imgs, min(n, len(imgs)))
    panels = []
    for p in sample:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        lp = label_path_for(p, split)
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                label = names.get(cid, str(cid))
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    im,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        panels.append(cv2.resize(im, (320, 240)))

    if not panels:
        return None
    # Simple grid
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    while len(panels) < rows * cols:
        panels.append(panels[-1] * 0)
    grid_rows = []
    for r in range(rows):
        grid_rows.append(cv2.hconcat(panels[r * cols : (r + 1) * cols]))
    grid = cv2.vconcat(grid_rows)
    out = AUDIT_DIR / f"label_audit_{split}.jpg"
    cv2.imwrite(str(out), grid)
    return out


def dir_checksum(path: Path) -> str:
    """Stable checksum over relative file paths + sizes (not full bytes — faster)."""
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for p in files:
        rel = str(p.relative_to(path)).encode()
        h.update(rel)
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def main() -> None:
    (ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "custom" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "custom" / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "custom" / "labels").mkdir(parents=True, exist_ok=True)

    download_and_extract()
    cfg = rewrite_data_yaml()
    # names may be int keys after yaml load
    names = {int(k): v for k, v in cfg["names"].items()}

    splits = ["train", "val", "test"]
    split_counts = {s: len(list_images(s)) for s in splits}
    class_counts = {s: dict(count_classes(s, names)) for s in splits}
    overlaps = check_filename_overlap(splits)
    phash_hits = perceptual_hash_collisions(splits)
    stats = {s: image_stats(s) for s in splits}

    for s in splits:
        out = save_audit_mosaic(s, names)
        if out:
            print(f"[prepare] Wrote label audit mosaic: {out}")

    # Flag rare classes on train
    train_counts = class_counts.get("train", {})
    total_boxes = sum(train_counts.values()) or 1
    rare = {
        k: v
        for k, v in sorted(train_counts.items(), key=lambda kv: kv[1])
        if v / total_boxes < 0.03
    }

    manifest = {
        "dataset": "construction-ppe",
        "source": DOWNLOAD_URL,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "path": str(DATA_DIR.resolve()),
        "content_fingerprint": dir_checksum(DATA_DIR),
        "split_image_counts": split_counts,
        "class_box_counts": class_counts,
        "rare_classes_train_lt_3pct": rare,
        "filename_overlaps": overlaps,
        "perceptual_hash_cross_split_collisions": phash_hits,
        "image_stats": stats,
        "class_names": names,
        "notes": [
            "Official Ultralytics Construction-PPE splits used as-is.",
            "No re-shuffle — keeps results comparable to others on this dataset.",
            "Review experiments/audit_samples/*.jpg before training.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prepare] Manifest → {MANIFEST}")
    print(f"[prepare] Split counts: {split_counts}")
    print(f"[prepare] Filename overlaps: {overlaps or 'none'}")
    print(f"[prepare] Cross-split near-dupes sampled: {len(phash_hits)}")
    if rare:
        print(f"[prepare] Rare train classes (<3% boxes): {rare}")
    print("[prepare] Done. Next: review docs/DATA_CARD.md then `make train-baseline`.")


if __name__ == "__main__":
    main()
