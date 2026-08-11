#!/usr/bin/env python3
"""
Download Hard Hats (HuggingFace / Roboflow mirror), convert COCO labels into
our Construction-PPE schema, and write configs/data_merged_hardhat.yaml.

Source classes → ours:
  hardhat     → helmet (0)
  no-hardhat  → no_helmet (7)

Only the external *train* split is added to our train set.
Our val/test stay Construction-PPE optimized (no leakage).
"""

from __future__ import annotations

import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hard-hat-workers"
DL = RAW / "_dl"
EXTRACTED = RAW / "extracted"
REMAPPED = ROOT / "data" / "hard-hat-remapped"
REPORT = ROOT / "experiments" / "hardhat_merge_report.json"
OUT_YAML = ROOT / "configs" / "data_merged_hardhat.yaml"
OPT_YAML = ROOT / "configs" / "data_optimized.yaml"

TRAIN_ZIP_URL = (
    "https://huggingface.co/datasets/keremberke/hard-hat-detection/"
    "resolve/main/data/train.zip"
)

# COCO category name (lower) → target class id in Construction-PPE
COCO_NAME_TO_TARGET = {
    "hardhat": 0,  # helmet
    "hard-hat": 0,
    "helmet": 0,
    "no-hardhat": 7,  # no_helmet
    "no_hardhat": 7,
    "nohardhat": 7,
    "head": 7,
}


def download_train_zip() -> Path:
    DL.mkdir(parents=True, exist_ok=True)
    dest = DL / "train.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[merge] Using cached {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"[merge] Downloading {TRAIN_ZIP_URL}")
    urllib.request.urlretrieve(TRAIN_ZIP_URL, dest)
    print(f"[merge] Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract_zip(zip_path: Path) -> Path:
    if EXTRACTED.exists():
        shutil.rmtree(EXTRACTED)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(EXTRACTED)
    return EXTRACTED


def find_coco_json(root: Path) -> Path:
    matches = list(root.rglob("_annotations.coco.json")) + list(root.rglob("*.coco.json"))
    if not matches:
        # any annotations json
        matches = [p for p in root.rglob("*.json") if "annot" in p.name.lower()]
    if not matches:
        raise SystemExit("[merge] No COCO annotation JSON found in zip")
    return matches[0]


def coco_to_yolo_line(bbox: list[float], w: int, h: int, cls_id: int) -> str | None:
    # COCO bbox: [x_min, y_min, width, height] absolute
    if w <= 0 or h <= 0:
        return None
    x, y, bw, bh = [float(v) for v in bbox]
    if bw <= 1 or bh <= 1:
        return None
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    # clamp
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def materialize_from_coco(coco_path: Path, max_images: int, seed: int) -> dict:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    cat_remap: dict[int, int] = {}
    for cid, cname in cats.items():
        key = cname.strip().lower().replace(" ", "-")
        tid = COCO_NAME_TO_TARGET.get(key)
        if tid is None:
            key2 = key.replace("-", "_")
            tid = COCO_NAME_TO_TARGET.get(key2)
        if tid is None:
            print(f"[merge] Dropping unmapped COCO class {cid}:{cname}")
            continue
        cat_remap[cid] = tid
    if not cat_remap:
        raise SystemExit(f"[merge] No usable COCO categories in {cats}")

    print(f"[merge] COCO categories: {cats}")
    print(f"[merge] Remap → target ids: { {cats[k]: v for k,v in cat_remap.items()} }")

    images = {int(im["id"]): im for im in data.get("images", [])}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    # Prefer images that contain no-hardhat
    prefer, other = [], []
    for iid, im in images.items():
        has_no = any(cat_remap.get(int(a["category_id"])) == 7 for a in anns_by_image.get(iid, []))
        (prefer if has_no else other).append(iid)

    rng = random.Random(seed)
    rng.shuffle(prefer)
    rng.shuffle(other)
    ordered = prefer + other
    if max_images > 0:
        ordered = ordered[:max_images]
    print(f"[merge] Using {len(ordered)} / {len(images)} images (prefer no-hardhat first)")

    if REMAPPED.exists():
        shutil.rmtree(REMAPPED)
    out_img = REMAPPED / "images" / "train"
    out_lbl = REMAPPED / "labels" / "train"
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    root = coco_path.parent
    box_counts: Counter = Counter()
    kept = 0
    missing_files = 0
    skipped_empty = 0

    for iid in ordered:
        im = images[iid]
        file_name = im["file_name"]
        src = root / file_name
        if not src.exists():
            # sometimes nested
            alt = next(root.rglob(Path(file_name).name), None)
            if alt is None:
                missing_files += 1
                continue
            src = alt
        w = int(im.get("width") or 0)
        h = int(im.get("height") or 0)
        lines = []
        for ann in anns_by_image.get(iid, []):
            tid = cat_remap.get(int(ann["category_id"]))
            if tid is None:
                continue
            line = coco_to_yolo_line(ann["bbox"], w, h, tid)
            if line:
                lines.append(line)
                box_counts[tid] += 1
        if not lines:
            skipped_empty += 1
            continue
        stem = f"hh_{Path(file_name).stem}"
        # sanitize
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        dst_img = out_img / f"{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst_img)
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        kept += 1

    named = {0: "helmet", 7: "no_helmet"}
    return {
        "images_kept": kept,
        "missing_files": missing_files,
        "skipped_empty_after_remap": skipped_empty,
        "box_counts": {str(k): int(v) for k, v in sorted(box_counts.items())},
        "box_counts_named": {named.get(k, str(k)): int(v) for k, v in sorted(box_counts.items())},
        "source_categories": cats,
        "cat_remap": {cats[k]: v for k, v in cat_remap.items()},
    }


def write_merged_yaml() -> None:
    opt = yaml.safe_load(OPT_YAML.read_text(encoding="utf-8"))
    opt_path = Path(opt["path"])
    if not opt_path.is_absolute():
        opt_path = (ROOT / opt_path).resolve()

    merged = {
        "path": str(ROOT / "data"),
        "train": [
            str((opt_path / "images" / "train").resolve()),
            str((REMAPPED / "images" / "train").resolve()),
        ],
        "val": str((opt_path / "images" / "val").resolve()),
        "test": str((opt_path / "images" / "test").resolve()),
        "names": {
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
        },
        "nc": 11,
    }
    OUT_YAML.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    print(f"[merge] Wrote {OUT_YAML}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge Hard Hat dataset into train")
    parser.add_argument(
        "--max-images",
        type=int,
        default=2500,
        help="Cap remapped train images (0 = all). Prefers no-hardhat.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not OPT_YAML.exists():
        raise SystemExit("[merge] Run `make optimize` first (need data_optimized.yaml)")

    zip_path = DL / "train.zip" if args.skip_download else download_train_zip()
    if not zip_path.exists():
        raise SystemExit(f"[merge] Missing {zip_path}")

    extract_zip(zip_path)
    coco_path = find_coco_json(EXTRACTED)
    print(f"[merge] COCO json: {coco_path}")
    stats = materialize_from_coco(coco_path, args.max_images, args.seed)
    write_merged_yaml()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": TRAIN_ZIP_URL,
        "license_note": "Hard Hats dataset via Roboflow/HF — CC BY 4.0 (verify before commercial redistribution)",
        "max_images": args.max_images,
        "remapped_dir": str(REMAPPED),
        "data_yaml": str(OUT_YAML),
        **stats,
        "note": "Val/test remain Construction-PPE optimized splits only.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[merge] Report → {REPORT}")
    print(f"[merge] Kept {stats['images_kept']} train images; boxes={stats['box_counts_named']}")


if __name__ == "__main__":
    main()
