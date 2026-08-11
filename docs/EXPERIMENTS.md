# Experiments

All primary training runs share: `imgsz=640`, `batch=16`, `device=mps`, `seed=42`, W&B project `ppe-safety-vision`, and **`configs/data_optimized.yaml`** (rare-class oversampled train; val/test unchanged) plus PPE-oriented augs (`copy_paste`, light `mixup`, `degrees`, stronger `scale`).

On Apple MPS, training uses `amp=false` and `workers=0` (see `scripts/train.py`) to avoid hard process kills.

**Rule:** change **one** primary knob per run. Select on **validation** metrics only (last logged epoch in `select_best.py`).

## Hyperparameter matrix

| Run | Config | Model | epochs | patience | lr0 | Purpose |
|---|---|---|---|---|---|---|
| A `baseline` | `train_baseline.yaml` | yolo11n | 100 | 20 | 0.01 | Reference |
| B `more_epochs` | `train_more_epochs.yaml` | yolo11n | 150 | 40 | 0.01 | Under-training check |
| C `lr_low` | `train_lr_low.yaml` | yolo11n | 100 | 20 | 0.005 (SGD) | Stabler LR — uses SGD so `lr0` is honored |
| D `model_s` | `train_model_s.yaml` | yolo11s | 100 | 20 | 0.01 | More capacity |
| E `with_custom` | `train_with_custom.yaml` | (usually winner recipe) | 80 | 20 | 0.01 | Add own labels |
| F `hardhat_ft` | `train_hardhat_ft.yaml` | fine-tune `weights/best.pt` | 60 | 20 | 0.001 | Extra Hard Hat remapped train data |

## How to run

```bash
export WANDB_PROJECT=ppe-safety-vision
wandb login

make prepare
make optimize
make train-all    # A→D
make select       # writes experiments/best_run.json
make eval
```

Optional Hard Hat merge experiment:

```bash
make merge-hardhat
make train-hardhat
make select
make eval
```

## Results table (val, last logged epoch)

Source: `runs/detect/<run>/results.csv` and `experiments/best_run.json`.

| Run | Epochs logged | Val mAP50 | Val mAP50-95 | Notes |
|---|---|---|---|---|
| baseline | 77 (early stop) | 0.549 | 0.260 | Reference |
| **more_epochs** | 150 | 0.546 | **0.269** | **Winner** |
| lr_low | 100 | 0.538 | 0.263 | Did not beat B |
| model_s | 82 (early stop) | 0.534 | 0.263 | Did not beat B |
| hardhat_ft | 21 (early stop) | 0.538 | 0.257 | Worse overall than B; not selected |
| with_custom | — | — | — | Not run (no custom site labels yet) |

Primary selection metric: **val mAP50-95** (tie-break: mAP50).

Winner recorded in:

```text
experiments/best_run.json
```

## Winner rationale

**`more_epochs`** won with val mAP50-95 **0.269** vs baseline **0.260**, lr_low **0.263**, and model_s **0.263**.

Longer training + higher patience on the same `yolo11n` recipe edged out both a lower-LR SGD trial and a larger `yolo11s` capacity run. Gains were modest (~0.01 mAP50-95), consistent with a data-limited problem rather than an under-tuned optimizer.

`hardhat_ft` (merged public Hard Hat Workers remapped into train) did **not** beat `more_epochs` on overall val/test mAP and was not selected for `weights/best.pt`.

## Held-out test (winner only)

After selection, `more_epochs` on the public test split (`experiments/test_eval_metrics.json`):

| Metric | Value |
|---|---|
| mAP50 | 0.526 |
| mAP50-95 | 0.264 |
| Precision | 0.678 |
| Recall | 0.526 |

See [MODEL_CARD.md](MODEL_CARD.md) for per-class detail. Violation classes remain weak; next gains require more labeled missing-PPE data.

## What we are not doing

- Random 20+ retries with no hypothesis
- Tuning on the test set
- Changing multiple knobs in one run (makes attribution impossible)

## Artifacts per run

Under `runs/detect/<name>/`:

- `weights/best.pt`, `weights/last.pt`
- `results.csv`, curves, confusion matrix plots
- W&B run (if logged in)
