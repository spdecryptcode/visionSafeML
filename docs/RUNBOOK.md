# Runbook

Operational guide for training, evaluating, serving, and retraining.

## 0. Environment (once)

```bash
cd "/Users/taranjotkaur/AI Projects/ppe-safety-vision"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add WANDB_API_KEY, then:
wandb login

python -c "import torch; print(torch.backends.mps.is_available())"
```

## 1. Prepare data

```bash
make prepare
make optimize
```

Checks:

1. Open `experiments/audit_samples/*.jpg`
2. Skim `experiments/data_manifest.json` and `experiments/data_optimize_report.json`
3. Keep [DATA_CARD.md](DATA_CARD.md) aligned with audit / optimize reports

## 2. Train experiments

```bash
# Fast path — one run
make train-baseline

# Full matrix A–D (longer)
make train-all
```

Optional Hard Hat merge fine-tune (usually does not beat `more_epochs`):

```bash
make merge-hardhat
make train-hardhat
```

If W&B is unavailable:

```bash
export WANDB_MODE=offline
# or
export WANDB_MODE=disabled
```

## 3. Select winner

```bash
make select
```

Produces:

- `experiments/best_run.json`
- `weights/best.pt`
- updated `configs/infer.yaml` model path

Current winner: **`more_epochs`**. Keep [EXPERIMENTS.md](EXPERIMENTS.md) in sync after new runs.

## 4. Test evaluation (once)

```bash
make eval
```

Metrics live in `experiments/test_eval_metrics.json` and [MODEL_CARD.md](MODEL_CARD.md).

## 5. Tune confidence on val

```bash
make tune
```

Writes best `conf` into `configs/infer.yaml` and `experiments/threshold_sweep.json`.

## 6. Export ONNX

```bash
make export
```

Artifact: `weights/best.onnx`.

## 7. Serve

```bash
make serve
# health
curl -s localhost:8000/health
# predict
curl -s -X POST localhost:8000/predict -F "file=@/path/to.jpg"
```

QA UI:

```bash
make demo
```

## 8. Custom images loop

```bash
# after Label Studio export into data/custom/raw/
make ingest-custom
make eval-custom          # score current winner on your holdout
make train-custom         # Run E
make select               # include with_custom if desired:
python scripts/select_best.py --runs baseline more_epochs lr_low model_s with_custom
make eval
make tune
make export
```

## 9. Tests

```bash
make test
```

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `MPS not available` | Falls back to CPU automatically; confirm `torch` macOS wheel |
| Train OOM / bus error | Lower `batch` in the train YAML (try 8) |
| `Model not found` on API | Run `make select` or set `configs/infer.yaml` model path |
| Dataset missing | `make prepare` |
| W&B login errors | `WANDB_MODE=offline` or `disabled` |
| Ultralytics download elsewhere | `prepare_data.py` uses the official zip into `data/construction-ppe` |

## Retrain on a client site (checklist)

1. Collect site images (varied cameras, times of day).
2. Label with same class schema ([LABELING_GUIDE.md](LABELING_GUIDE.md)).
3. Ingest / point data YAML at new dataset.
4. Run train matrix (or baseline + 1–2 targeted runs).
5. Select on **their** val split; evaluate on their holdout test.
6. Tune conf; export; redeploy API.
7. Document domain gap vs Construction-PPE in the model card.
