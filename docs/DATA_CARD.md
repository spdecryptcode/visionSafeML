# Data Card

## Datasets

### 1. Construction-PPE (public baseline)

| Field | Value |
|---|---|
| Name | Ultralytics Construction-PPE |
| Task | Object detection (PPE + missing PPE + person) |
| Approx size | ~178 MB / ~1,416 images |
| Official splits | train 1,132 · val 143 · test 141 |

#### Classes (official order)

| ID | Name | Role |
|---|---|---|
| 0 | helmet | worn |
| 1 | gloves | worn |
| 2 | vest | worn |
| 3 | boots | worn |
| 4 | goggles | worn |
| 5 | none | other |
| 6 | Person | person |
| 7 | no_helmet | **violation** |
| 8 | no_goggle | **violation** |
| 9 | no_gloves | **violation** |
| 10 | no_boots | **violation** |

> Note: there is no dedicated `no_vest` class in this dataset.

### 2. Hard Hats (optional public merge)

| Field | Value |
|---|---|
| Name | Hard Hats (Roboflow → HuggingFace mirror `keremberke/hard-hat-detection`) |
| Task | hardhat / no-hardhat detection |
| Ingest | `make merge-hardhat` → `scripts/merge_hardhat.py` |
| Remap | `hardhat`→`helmet` (0), `no-hardhat`→`no_helmet` (7) |
| Used in train | Up to 2,500 train images preferred for `no-hardhat` (see `experiments/hardhat_merge_report.json`) |
| Val / test | **Not** mixed in — Construction-PPE val/test stay clean |
| Config | `configs/data_merged_hardhat.yaml` |
| Outcome | Fine-tune `hardhat_ft` did not beat `more_epochs` overall; kept as experiment only |

### 3. Custom site photos (your data)

| Field | Value |
|---|---|
| Name | Custom PPE photos |
| Owner | You (project author) |
| Target size | 50–100 labeled images (v1) |
| Schema | Same 11 classes as above |
| Raw drop | `data/custom/raw/images` + `data/custom/raw/labels` |
| After ingest | `data/custom/images/{train,holdout}` |
| Holdout | ~20% reserved for custom evaluation |
| Guide | [LABELING_GUIDE.md](LABELING_GUIDE.md) |
| Status | **Not collected yet** — primary path to improve violation recall |

---

## Audit checklist (fill after `make prepare`)

Run:

```bash
make prepare
cat experiments/data_manifest.json
open experiments/audit_samples/label_audit_train.jpg
```

Then record findings here:

### Split integrity

| Check | Result | Notes |
|---|---|---|
| Filename overlap train/val/test | **none** | clean |
| Perceptual-hash cross-split collisions | **0** (sampled) | no near-dupes found in sample |
| Official splits kept (no reshuffle) | Yes | by design |
| Split image counts | train **1132** / val **143** / test **141** | matches Ultralytics docs |

### Class balance (train box counts)

From `experiments/data_manifest.json` → `class_box_counts.train`:

```text
Person: 1770
helmet: 1341
vest: 1269
boots: 1235
gloves: 1146
none: 651
no_gloves: 442
goggles: 419
no_helmet: 400
no_goggle: 337
no_boots: 88
```

Rare classes (&lt;3% of train boxes):

```text
no_boots: 88  ← expect weaker recall; watch per-class metrics
```

### Image stats

See `experiments/data_manifest.json` → `image_stats` (sampled widths/heights per split).

### Label audit notes

Mosaics written to `experiments/audit_samples/`:

- [x] Audit mosaics generated for train/val/test
- [ ] Manually review mosaics before trusting production use
- Notes: Missing-PPE classes are present but imbalanced; `no_boots` is rare.

---

## Dataset optimization (public set, no new photos)

Run:

```bash
make optimize
```

Creates `data/construction-ppe-optimized/` + `configs/data_optimized.yaml`.

| Step | Action |
|---|---|
| Label hygiene | Skip train images with empty/missing labels |
| Rare-class balance | Oversample train images containing `no_boots` (primary) and lightly boost `no_goggle` |
| Val / test | **Untouched** — no leakage, metrics stay honest |
| Train augs (in train YAMLs) | `copy_paste`, light `mixup`, `degrees`, stronger `scale` |

Report: `experiments/data_optimize_report.json`  
Training configs A–D point at `configs/data_optimized.yaml`.

| Metric | Before | After |
|---|---|---|
| Train images | 1,132 | **1,688** (+556 oversampled) |
| `no_boots` boxes | 88 | **616** |

## How splits are used

| Split | Used for |
|---|---|
| Train (optimized) | Model learning only |
| Val | Compare runs A–D (+ optional F), pick winner, tune `conf` |
| Test | **One** final report after selection |
| Hard Hat remapped train | Optional extra train only (`make merge-hardhat`) |
| Custom holdout | Optional later site imagery |

Never tune hyperparameters or thresholds on the test set.

---

## Provenance fingerprint

After prepare, `experiments/data_manifest.json` stores `content_fingerprint` (path+size hash) so you can detect accidental dataset changes before retraining.
