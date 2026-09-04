---
title: Aqueous Solubility Predictor
emoji: 🧪
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Aqueous solubility predictor

Random forest on ECFP4 fingerprints + RDKit descriptors, trained on the
**scaffold-split** training half of ESOL/Delaney (1117 molecules after
deduplication).

Enter a SMILES string; you get a predicted log S (mol/L), the structure,
and the five nearest training molecules by Tanimoto similarity.

## Why the neighbor table is the important part

A prediction for a molecule unlike anything in training is extrapolation
dressed up as a number. This app names that: below 0.4 maximum Tanimoto
to any training molecule, it labels the result as outside the
applicability domain instead of returning a confident-looking value.

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
does not belong behind a web request.

## Data

Delaney, *J. Chem. Inf. Comput. Sci.* **44** (2004) 1000.
