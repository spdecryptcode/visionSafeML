# Model Card

> Metrics below are from real evaluation outputs under `experiments/*_metrics.json`.

## Model details

| Field | Value |
|---|---|
| Task | Multi-class PPE object detection |
| Framework | Ultralytics YOLO11 |
| Winner run | **`more_epochs`** (`configs/train_more_epochs.yaml`) |
| Architecture | YOLO11n (`yolo11n.pt` fine-tuned) |
| Weights | `weights/best.pt` ← `runs/detect/more_epochs/weights/best.pt` |
| ONNX | `weights/best.onnx` (after `make export`) |
| Inference conf | `0.25` in `configs/infer.yaml` (default; run `make tune` to sweep) |
| Device (dev) | Apple MPS (M3 Max); AMP disabled for MPS train stability |

## Intended use

- Detect worn PPE and **missing** PPE on construction-site still images.
- Produce structured violation codes for downstream compliance workflows.
- Apply **association rules** in `src/violations.py`: if a `Person` has no nearby `helmet` / `vest`, emit `missing_helmet` / `missing_vest` even when the `no_*` detector is weak.
- Prototype / internal evaluation; retrain on site-specific labeled images before production use on a new site.

## Out of scope

- Certified safety system / sole reliance for life-critical decisions
- Live multi-camera RTSP edge deployment (v1)
- Guaranteeing performance on sites with very different cameras, PPE styles, or lighting without retraining

## Training data

- Primary: Ultralytics Construction-PPE optimized train set (see [DATA_CARD.md](DATA_CARD.md))
- Optional experiment: remapped Hard Hat Workers train images (`make merge-hardhat`) — **not** used in the selected winner
- Additional: author-collected custom labeled images (after ingest) — not yet collected

## Evaluation results

### Public test split (winner: more_epochs)

Source file: `experiments/test_eval_metrics.json`

| Metric | Value |
|---|---|
| mAP50 | **0.526** |
| mAP50-95 | **0.264** |
| Precision | **0.678** |
| Recall | **0.526** |

#### Per-class (focus on violations + key worn PPE)

| Class | Precision | Recall | AP50 |
|---|---|---|---|
| no_helmet | 0.235 | 0.215 | 0.170 |
| no_goggle | 0.304 | 0.242 | 0.222 |
| no_gloves | 0.380 | 0.121 | 0.134 |
| no_boots | 1.000 | **0.000** | ~0.000 |
| helmet | 0.930 | 0.896 | 0.933 |
| vest | 0.807 | 0.860 | 0.880 |
| Person | 0.836 | 0.809 | 0.814 |

Worn PPE and person detection are relatively strong; **missing-PPE classes are weak**, especially `no_boots`.

### Hard Hat fine-tune (not selected)

Source: `experiments/test_eval_hardhat_ft_direct_metrics.json`

| Metric | more_epochs (winner) | hardhat_ft |
|---|---|---|
| Test mAP50 | 0.526 | 0.495 |
| Test mAP50-95 | 0.264 | 0.246 |
| no_helmet recall | 0.215 | 0.250 |

Slight `no_helmet` recall gain, worse overall mAP → kept `more_epochs`.

### Custom holdout (own images)

Source file: `experiments/custom_holdout_eval_metrics.json`

| Metric | Value |
|---|---|
| mAP50 | _not run_ |
| mAP50-95 | _not run_ |
| Notes | No custom site labels ingested yet |

## Operating point

| Setting | Value | How chosen |
|---|---|---|
| conf | 0.25 | default (threshold sweep not yet written into infer.yaml) |
| iou | 0.45 | default NMS |
| imgsz | 640 | train/infer match |

## Failure modes

Known / observed on this project:

- [x] Rare violation classes under-detected (`no_boots` recall ≈ 0 on test)
- [x] Weak `no_helmet` / `no_gloves` / `no_goggle` recall vs worn classes
- [x] Class imbalance in public train set (partially mitigated by oversampling)
- [ ] Occlusion (person partially behind equipment) — expected, not exhaustively audited
- [ ] Small / distant workers — expected
- [ ] Unusual hard-hat colors or non-standard PPE — expected on new sites
- [ ] Night / harsh backlight — expected on new sites

Mitigations in serving: person↔PPE association rules; next model gains need more labeled violation images.

## Ethical / privacy considerations

- Images may contain identifiable people — handle custom photos responsibly.
- System can produce false negatives (missed missing-PPE); do not treat as sole safety control.
- Disclose public training data provenance when presenting the project.

## How to reproduce reported metrics

```bash
make prepare
make optimize
make train-all
make select
make eval
# optional:
make tune
make export
```
