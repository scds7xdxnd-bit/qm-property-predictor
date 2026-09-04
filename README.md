# QM-augmented property predictor

**The question:** once a model already has Morgan fingerprints and cheap
RDKit descriptors, do quantum-chemical features — B3LYP orbital energies,
dipole moment, Mulliken charges — add anything measurable? Or is the
physics already implicit in the structure?

The answer is not obvious, which is what makes it worth running. Built as
a project across all seven chapters of *파이썬을 이용한 화학 인공지능*
(정근홍, 사이플러스, 2024).

---

## Chapter coverage

| Book chapter | Where it lives |
|---|---|
| Ch 1 — RDKit, descriptors, fingerprints, Tanimoto | `src/qmprop/features.py`, `data.py` |
| Ch 2 — chemical space, data assembly | `src/qmprop/data.py`, notebook §1 |
| Ch 3 — ridge → lasso → RF → SVM → XGBoost | `src/qmprop/models.py`, `scripts/04_train.py` |
| Ch 4 — neural network, benchmarked honestly against Ch 3 | `models.py` (`mlp`) |
| Ch 5 — quantum theory | `src/qmprop/qm.py` docstrings; the solvers are yours to write |
| Ch 6 — driving QC software from Python | `src/qmprop/qm.py`, `scripts/03_run_qm.py` |
| Ch 7 — LLM layer | `app/app.py` (extension point — see below) |

Two additions the book does not cover, both load-bearing:
**scaffold splitting** (`src/qmprop/splits.py`) and the
**applicability-domain check** in the app.

---

## Setup

```bash
conda env create -f environment.yml && conda activate qmprop
```

Or with pip (`pip install -r requirements.txt`). Conda is smoother —
PySCF in particular is happier from conda-forge.

> **Use Python 3.11–3.13.** RDKit 2026.03.6 on Python 3.14 raises
> `TypeError: No to_python converter found for C++ type:
> std::basic_string_view` on `import AllChem` — a Boost binding
> mismatch, not something you can work around in code. Verified
> working on 3.13; `environment.yml` pins 3.11.

## Run

```bash
make data       # download ESOL, canonicalize, dedupe        ~10 s
make features   # 2048-bit ECFP4 + ~200 descriptors          ~30 s
make train      # Ch 3 baselines vs the Ch 4 network         ~2 min
make qm         # B3LYP/6-31G* single points                 SLOW
make ablation   # the actual question                        ~3 min
make app        # Gradio predictor
make test       # pytest
```

**Smoke-test the QM step before committing to it:**

```bash
python scripts/03_run_qm.py --limit 5
```

Each molecule is 5–60 s depending on size, so the default 200-molecule
subset is a coffee-to-overnight run. Results append to CSV after every
molecule — interrupt freely, rerunning resumes.

---

## Measured baselines

Actually run, not estimated. ESOL, 1117 molecules after deduplication
(11 duplicates merged), 2048-bit ECFP4 + 199 RDKit descriptors,
80/20 split, 224 test molecules.

| model | scaffold RMSE | random RMSE | inflation |
|---|---|---|---|
| XGBoost | **0.900** (R² 0.808) | 0.595 (R² 0.926) | **+51%** |
| Random forest | 0.909 (R² 0.804) | 0.621 (R² 0.920) | +46% |
| Ridge (RidgeCV) | 1.072 (R² 0.728) | 1.145 (R² 0.728) | — |
| MLP | 1.311 (R² 0.593) | 0.924 (R² 0.823) | +42% |

Two things fall out of this table.

**The random split inflates scores by roughly half an RMSE unit.** If you
read a solubility paper reporting ~0.6 RMSE on ESOL without saying how it
split, that is very likely this artifact rather than a better model.

**XGBoost beats the neural network, decisively.** 0.900 against 1.311 on
the honest split — which is the Ch 3 versus Ch 4 comparison the book sets
up but does not close. At ~1100 molecules there is not enough data for the
network to earn its capacity. This is the single most useful thing to
know before spending a week on Ch 4.

A footnote on the ridge row: with a fixed `alpha=1.0` it scored RMSE 2.141
and **R² −0.087** — worse than predicting the mean. At 2247 features on
893 samples a fixed penalty is badly under-regularized. `models.py` uses
`RidgeCV`/`LassoCV`/`ElasticNetCV` so the linear arm is a real baseline
rather than a strawman.

---

## What the ablation actually compares

Five arms, all scored on **exactly the same molecules and the same
scaffold split** — the subset for which QM converged:

| Arm | Features |
|---|---|
| `fingerprint` | ECFP4 only |
| `descriptors` | RDKit descriptors only |
| `fingerprint+desc` | both — the baseline to beat |
| `qm only` | 8 quantum features alone |
| `fingerprint+desc+qm` | everything |

The same-subset constraint is the design decision that makes the result
mean anything. Comparing a QM arm on 180 small molecules against a
fingerprint arm on all 1117 measures the subset, not the features — and
that mistake is common enough in published work to be worth naming.

`05_ablation.py` prints the per-model RMSE delta and, next to it, the
noise floor for that test-set size. **A delta smaller than the noise
floor is "no measurable effect", not a win.** Report it that way.

### What to expect

The QM arm has not been run at scale yet — that is the open question and
the reason the project exists. Plausible outcomes:

- **QM adds little.** Likely for solubility — it is dominated by polarity
  and H-bonding, which TPSA and LogP already capture. A null result here
  is a real finding, and worth writing up.
- **QM helps the linear models but not the trees.** Would suggest the
  trees were already extracting equivalent information from structure.
- **QM helps everything.** Would be the interesting case; check first
  that molecule size did not leak in as a confound, since both QM cost
  and solubility correlate with it.

Swap the target to something more electronic — HOMO–LUMO gap prediction,
redox potentials, reaction barriers — and the QM arm should look much
better. That contrast is itself a good second experiment.

---

## QM module, verified

`qm_descriptors()` on ethanol, B3LYP/6-31G* single point over an
MMFF-relaxed ETKDG conformer — 2.6 s, geometry embedding 0.03 s:

| quantity | computed | check |
|---|---|---|
| HOMO | −7.077 eV | plausible for a saturated alcohol |
| LUMO | +2.059 eV | positive, as expected for a bound virtual orbital |
| gap | 9.135 eV | wide, consistent with a colorless insulator |
| dipole | 1.659 D | **experimental gas-phase value is 1.69 D** |
| Mulliken | −0.626 … +0.391 | O most negative, its H most positive |

The dipole landing within 0.03 D of experiment is the useful check: it
says the geometry, the basis, and the unit conversion are all correct.
A wrong Angstrom/Bohr conversion or a broken conformer would miss by a
factor, not by 2%.

## Honest limitations

- **Single points on MMFF geometries**, not DFT-optimized structures.
  ~50× cheaper, fine for descriptors, not publishable as energetics.
- **Kohn–Sham orbital energies are not ionization potentials.** They
  correlate with reactivity, which is all a feature needs to do.
- **One conformer per molecule.** Flexible molecules deserve a Boltzmann
  ensemble; that is a real extension, not a footnote.
- **n ≈ 200 for the QM arm.** Small. Widen `qm.subset_size` before
  drawing strong conclusions.
- **ESOL is small and old** (1128 molecules, curated 2004). Good for a
  first pass; move to AqSolDB (~10k) if the result looks interesting.

---

## Extension points

**Ch 4 → graph neural networks.** `build_model()` is the only function
that needs to change. Fingerprints throw away connectivity a GNN keeps.

**Ch 7 → an agent layer.** The modern version of the ChatGPT chapter is
not "ask a model a question" but giving an LLM `qm_descriptors()` and
`predict()` as tools, so it resolves a name to SMILES, runs the pipeline,
and explains the answer using the nearest-neighbor table. Validate every
SMILES it produces through `Chem.MolFromSmiles()` first — `None` means
the molecule was never real.

**Better QM.** Geometry optimization instead of single points; a
conformer ensemble; ωB97X-D for dispersion.

---

## Layout

```
config.yaml           every knob, one file
src/qmprop/
  data.py             download, canonicalize, dedupe by InChIKey
  features.py         descriptors + ECFP, assembled into blocks
  splits.py           scaffold split (the correction that matters)
  qm.py               RDKit geometry -> PySCF -> 8 QM features
  models.py           ridge/lasso/SVR/RF/XGBoost/MLP registry
  evaluate.py         metrics, parity plots, text tables
scripts/01..05        the pipeline, in order
notebooks/            Colab quickstart
app/app.py            Gradio predictor with applicability-domain warning
tests/                pytest — splits and features
```

## Data

ESOL / Delaney: 1128 compounds with measured aqueous solubility, via
DeepChem's mirror. Delaney, *J. Chem. Inf. Comput. Sci.* 44 (2004) 1000.
