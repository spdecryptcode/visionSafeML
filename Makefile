PYTHON ?= python3
export PYTHONPATH := .

.PHONY: help setup prepare optimize merge-hardhat train-baseline train-all train-hardhat train-custom select eval tune export serve demo test clean

help:
	@echo "PPE Safety Vision — common targets"
	@echo "  make setup           Create venv and install requirements"
	@echo "  make prepare         Download + audit Construction-PPE"
	@echo "  make optimize        Rare-class oversample + cleaned train set"
	@echo "  make merge-hardhat   Remap Hard Hat Workers → train (helmet/no_helmet)"
	@echo "  make train-baseline  Run A only"
	@echo "  make train-all       Runs A–D (hyperparameter matrix)"
	@echo "  make train-hardhat   Fine-tune best.pt on merged hardhat data"
	@echo "  make select          Pick best by val mAP → weights/best.pt"
	@echo "  make eval            Held-out test evaluation"
	@echo "  make tune            Sweep conf on val → configs/infer.yaml"
	@echo "  make export          Export ONNX"
	@echo "  make train-custom    Run E after ingesting your labeled photos"
	@echo "  make serve           FastAPI on :8000"
	@echo "  make demo            Gradio QA UI on :7860"
	@echo "  make test            Unit + API smoke tests"

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate: source .venv/bin/activate"
	@echo "Then: wandb login   (optional but recommended)"

prepare:
	$(PYTHON) scripts/prepare_data.py

optimize:
	$(PYTHON) scripts/optimize_data.py

merge-hardhat:
	$(PYTHON) scripts/merge_hardhat.py

train-hardhat:
	$(PYTHON) scripts/train.py --config configs/train_hardhat_ft.yaml

train-baseline:
	$(PYTHON) scripts/train.py --config configs/train_baseline.yaml

train-all: train-baseline
	$(PYTHON) scripts/train.py --config configs/train_more_epochs.yaml
	$(PYTHON) scripts/train.py --config configs/train_lr_low.yaml
	$(PYTHON) scripts/train.py --config configs/train_model_s.yaml

select:
	$(PYTHON) scripts/select_best.py

eval:
	$(PYTHON) scripts/evaluate.py --split test --name test_eval

eval-val:
	$(PYTHON) scripts/evaluate.py --split val --name val_eval

tune:
	$(PYTHON) scripts/tune_threshold.py

export:
	$(PYTHON) scripts/export_onnx.py

ingest-custom:
	$(PYTHON) scripts/ingest_custom.py

train-custom:
	$(PYTHON) scripts/train.py --config configs/train_with_custom.yaml

eval-custom:
	$(PYTHON) scripts/evaluate.py --data configs/data_custom_holdout.yaml --split test --name custom_holdout_eval

serve:
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000

demo:
	$(PYTHON) app/demo.py

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf runs/ weights/*.pt weights/*.onnx experiments/audit_samples .pytest_cache
	@echo "Kept data/ and configs/. Remove data/construction-ppe manually if needed."
