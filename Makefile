.PHONY: help setup data features qm train ablation app test clean

help:
	@echo "make setup     conda env from environment.yml"
	@echo "make data      download + clean ESOL          (~10 s)"
	@echo "make features  descriptors + fingerprints     (~30 s)"
	@echo "make qm        B3LYP/6-31G* on the subset     (slow: minutes to hours)"
	@echo "make train     Ch3 vs Ch4 baselines           (~2 min)"
	@echo "make ablation  does QM help?                  (~3 min, needs qm)"
	@echo "make app       launch the Gradio predictor"
	@echo "make test      pytest"

setup:
	conda env create -f environment.yml

data:
	python scripts/01_fetch_data.py

features: data
	python scripts/02_featurize.py

qm: data
	python scripts/03_run_qm.py

train: features
	python scripts/04_train.py

ablation: features
	python scripts/05_ablation.py

app:
	python app/app.py

test:
	pytest -q tests/

clean:
	rm -rf results/*.json results/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
