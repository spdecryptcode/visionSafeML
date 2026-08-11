# visionSafeML — Workplace PPE Compliance Vision

Enterprise computer-vision platform for **construction-site PPE compliance monitoring**.

visionSafeML detects worn and missing personal protective equipment—including helmets, vests, gloves, boots, and goggles—selects the highest-performing model from governed training experiments, and delivers structured violation findings through a production FastAPI inference service.

The platform is engineered for on-premises and workstation deployment.

---

## Business problem

Manual PPE inspections do not scale across multi-crew construction environments. visionSafeML provides automated visual compliance screening from site imagery so safety and operations teams can identify missing protective equipment, reduce inspection latency, and maintain an auditable detection workflow.

## Solution

| Capability | Implementation |
|---|---|
| Detection | YOLO11n fine-tuned on labeled construction-site PPE imagery |
| Current winner | `more_epochs` — test mAP50 **0.526**, mAP50-95 **0.264** |
| Compliance output | Structured codes (`missing_helmet`, `missing_gloves`, …) + person/PPE association rules |
| Experiment tracking | Weights & Biases across controlled training runs |
| Model selection | Best checkpoint by validation mAP50-95 |
| Evaluation | Held-out test metrics + optional site-image holdout |
| Serving | FastAPI inference service + Gradio QA console |
| Deployment artifact | ONNX export alongside PyTorch weights |
| Runtime | Apple Silicon (MPS) supported for local training and inference |

Pipeline: **data audit → optimize → multi-run training → select → evaluate → threshold tune → serve**.

---

## System docs

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, API contract |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | Datasets, licenses, splits, audit results |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Metrics, failure modes, intended use |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Training matrix and winner selection |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operate, retrain, and release checklist |
| [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) | Site image labeling standard (11-class schema) |

---

## Environment setup

```bash
git clone https://github.com/spdecryptcode/visionSafeML.git
cd visionSafeML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set WANDB_API_KEY, then:
wandb login
```

Confirm MPS (Apple Silicon):

```bash
python -c "import torch; print('mps', torch.backends.mps.is_available())"
```

---

## Operations

```bash
make prepare          # dataset pull + integrity / class audit
make optimize         # rare-class oversample (val/test untouched)
make train-all        # training runs A–D (or make train-baseline)
make select           # pick winner → weights/best.pt
make eval             # held-out test evaluation
make tune             # confidence threshold on validation
make export           # ONNX artifact
make test             # unit + API smoke tests
make serve            # API → http://localhost:8000
make demo             # QA UI → http://localhost:7860
```

Optional Hard Hat merge experiment: `make merge-hardhat` then `make train-hardhat`.

### Training runs

| Run | Config | Change under test | Val mAP50-95 |
|---|---|---|---|
| A `baseline` | `configs/train_baseline.yaml` | yolo11n, lr 0.01, 100 epochs | 0.260 |
| B `more_epochs` | `configs/train_more_epochs.yaml` | longer training / patience | **0.269** (winner) |
| C `lr_low` | `configs/train_lr_low.yaml` | lower LR (SGD) | 0.263 |
| D `model_s` | `configs/train_model_s.yaml` | yolo11s capacity | 0.263 |
| E `with_custom` | `configs/train_with_custom.yaml` | include site-labeled images | not run |
| F `hardhat_ft` | `configs/train_hardhat_ft.yaml` | + remapped Hard Hat train | 0.257 (not selected) |

Winner is selected on **validation** metrics only. The test set is evaluated once after selection. Details: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

### Site imagery (production data loop)

1. Capture site photos under the labeling standard in [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md).
2. Export YOLO labels into `data/custom/raw/`.
3. Ingest, evaluate holdout, and retrain:

```bash
make ingest-custom
make eval-custom
make train-custom
make select
make eval
make tune
make export
```

---

## Inference API

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/predict \
  -F "file=@/path/to/site.jpg" | jq '.summary'
```

Example summary payload:

```json
{
  "compliant": false,
  "violation_codes": ["missing_helmet"],
  "worn_ppe": ["vest"],
  "person_count": 1
}
```

Full contract: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Detected classes

Worn PPE: `helmet`, `gloves`, `vest`, `boots`, `goggles`  
Violations: `no_helmet`, `no_gloves`, `no_boots`, `no_goggle`  
Context: `Person`, `none`

---

## License notes (non-public use)

visionSafeML application code, configs, documentation, and custom operational artifacts are **proprietary and confidential**. See [LICENSE](LICENSE).

| Asset | Terms |
|---|---|
| visionSafeML source, configs, docs, runbooks | Non-public. Internal / authorized client use only. No redistribution without written permission. |
| Trained weights produced for a client engagement | Client-confidential unless otherwise agreed in writing. |
| Ultralytics runtime / Construction-PPE distribution | Third-party terms (**AGPL-3.0** for Ultralytics distribution). Those licenses are independent and must be reviewed before any redistribution of derived model assets. |

Unauthorized copying, public release, or resale of this project is prohibited.
