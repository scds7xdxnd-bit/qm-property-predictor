---
title: Solubility Predictor
emoji: 🧪
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
---

# Aqueous solubility predictor

XGBoost on ECFP4 fingerprints + RDKit descriptors, trained on the
**scaffold-split** training half of ESOL/Delaney (1117 molecules after
deduplication).

Type a **name** (`caffeine`, `aspirin`, `table salt`) or a **SMILES**
string. You get a predicted log S (mol/L), the structure, the five
nearest training molecules with their *measured* values, and the
features that moved this particular prediction.

## The number is the least interesting part

Three things travel with every prediction:

**Nearest neighbours, with measurements.** These are real data points you
can check the answer against. If the neighbours look nothing like your
molecule, the prediction is not trustworthy however confident it looks.

**Exact attribution.** The feature table is XGBoost's `pred_contribs` —
exact TreeSHAP, where contributions plus bias reproduce the prediction to
~10⁻⁶. It is arithmetic, not a story told alongside the model. A typical
prediction has several hundred non-zero contributions, so the table shows
the largest few and says so rather than implying it is the whole sum.

**A domain warning.** Below 0.4 maximum Tanimoto to anything in training,
the result is labelled extrapolation instead of being returned as a
confident-looking value. Try `table salt` — an ionic solid is exactly the
chemistry this model has never seen.

## How good is 0.90 RMSE?

Better than it sounds. Two independent curations of aqueous solubility
(ESOL and AqSolDB) disagree with each other by ~0.34 log units RMSE on
the same molecules, so that is roughly the floor — not zero. And 57% of
the scaffold-split test set sits below the 0.4 Tanimoto line this app
warns about, meaning the headline score is largely earned on the hard
cases rather than on near-duplicates.

## Why scaffold splitting

A random train/test split scatters near-identical analogs across both
sides, so the model is graded on molecules it has effectively already
seen. Splitting by Bemis–Murcko scaffold puts each chemical series
wholly on one side. The scores are worse and they are real.

## What this is part of

The front end of [qm-property-predictor](https://github.com/scds7xdxnd-bit/qm-property-predictor),
a project asking whether quantum-chemical features (B3LYP/6-31G* HOMO,
LUMO, dipole, Mulliken charges via PySCF) improve property prediction
once fingerprints and cheap descriptors are already in hand.

The Space runs the cheap half only — no PySCF, since a DFT calculation
does not belong behind a web request. Name resolution calls out to
PubChem, falling back to ChEMBL.

## Data

Delaney, *J. Chem. Inf. Comput. Sci.* **44** (2004) 1000.
