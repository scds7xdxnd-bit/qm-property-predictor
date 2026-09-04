.PHONY: help setup data features chemspace theory qm train gnn ablation explain scale app test all clean

help:
	@echo "Ch 2  make data       download + clean ESOL             (~10 s)"
	@echo "Ch 2  make chemspace  t-SNE/UMAP + cross-source noise   (~2 min)"
	@echo "Ch 1  make features   descriptors + fingerprints        (~30 s)"
	@echo "Ch 5  make theory     box/oscillator solvers, polyenes  (~20 s)"
	@echo "Ch 6  make qm         B3LYP/6-31G* on the subset        (SLOW: hours)"
	@echo "Ch 3  make train      ridge / RF / XGBoost baselines    (~2 min)"
	@echo "Ch 4  make gnn        message-passing GNN               (~10 min)"
	@echo "Ch 6  make ablation   does QM actually help?            (needs qm)"
	@echo "Ch 7  make explain Q='caffeine'   explained prediction"
	@echo "Ch 2+4 make scale     does 9x more data change the verdict? (~1 h)"
	@echo "      make app        launch the Gradio predictor"
	@echo "      make test       pytest"
	@echo "      make all        everything except qm and ablation"

setup:
	conda env create -f environment.yml

data:
	python scripts/01_fetch_data.py

features: data
	python scripts/02_featurize.py

chemspace: data
	python scripts/02b_chemspace.py

theory:
	python scripts/06_theory.py

# Parallel across molecules, one thread each. Tune --workers to your box:
# more is not always faster, and on a memory-tight machine it is slower.
qm: data
	python scripts/03_run_qm.py

train: features
	python scripts/04_train.py

gnn: features
	python scripts/08_gnn.py

ablation: features
	python scripts/05_ablation.py

Q ?= caffeine
explain: features
	python scripts/07_explain.py "$(Q)"

# Tests the README's own claim that the GNN loses for want of data.
scale: data
	python scripts/09_scale.py

app:
	python app/app.py

test:
	pytest -q tests/

all: data features chemspace theory train gnn
	@echo
	@echo "Done. The one thing left is the expensive arm:"
	@echo "  make qm && make ablation"

clean:
	rm -rf results/*.json results/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
