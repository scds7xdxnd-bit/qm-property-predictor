# QM-augmented property predictor

**Live demo:** https://huggingface.co/spaces/taeyangkimi/solubility-predictor


**The question:** once a model already has Morgan fingerprints and cheap
RDKit descriptors, do quantum-chemical features — B3LYP orbital energies,
dipole moment, Mulliken charges — add anything measurable? Or is the
physics already implicit in the structure?

**The answer, measured: no.** Across 195 molecules with converged DFT,
scaffold k-fold and a paired bootstrap, QM features move RMSE by less
than the confidence interval for three of four models. The fourth
appears to improve until you replace the quantum features with Gaussian
noise and get most of the same gain. [Full result below](#the-answer-no-qm-does-not-help-here).

Built as a project across all seven chapters of *파이썬을 이용한 화학 인공지능*
(정근홍, 사이플러스, 2024).

---

## Chapter coverage

| Book chapter | Where it lives |
|---|---|
| Ch 1 — RDKit, descriptors, fingerprints, Tanimoto | `src/qmprop/features.py`, `data.py` |
| Ch 2 — chemical space, data assembly | `src/qmprop/data.py`, `external.py`, `scripts/02b_chemspace.py` |
| Ch 3 — ridge → lasso → RF → SVM → XGBoost | `src/qmprop/models.py`, `scripts/04_train.py` |
| Ch 4 — neural network **and a graph network** vs Ch 3 | `models.py` (`mlp`), `src/qmprop/gnn.py`, `scripts/08_gnn.py` |
| Ch 5 — quantum theory, solved from scratch | `src/qmprop/theory.py`, `scripts/06_theory.py` |
| Ch 6 — driving QC software from Python | `src/qmprop/qm.py`, `scripts/03_run_qm.py` |
| Ch 7 — name → structure → prediction → explanation | `src/qmprop/explain.py`, `scripts/07_explain.py`, `app/app.py` |

Three additions the book does not cover, all load-bearing:
**scaffold splitting** (`src/qmprop/splits.py`), the
**applicability-domain check** in the app, and a **measured noise floor**
from an independent curation (below).

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
make chemspace  # t-SNE/UMAP + cross-source noise floor      ~2 min
make theory     # Ch 5 solvers, and where they break         ~20 s
make train      # Ch 3 baselines vs the Ch 4 network         ~2 min
make gnn        # Ch 4 stretch: message-passing GNN          ~10 min
make qm         # B3LYP/6-31G* single points                 SLOW
make ablation   # the actual question                        ~3 min
make explain Q='caffeine'   # Ch 7, explained prediction
make app        # Gradio predictor
make test       # pytest (102 tests)
make all        # everything except qm and ablation
```

`make qm` picks its worker count from **RAM, not cores** — roughly 2 GB
per worker with 2 GB reserved for the OS. That is not fussiness: seven
workers on an 8 GB machine drove it into swap, worker CPU collapsed to
~28% each while `kernel_task` burned a full core on memory compression,
and throughput was *worse* than with three. Override with `--workers N`
if you know your box.

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
| GNN (message-passing) | 1.021 (R² 0.753) | — | — |
| Ridge (RidgeCV) | 1.072 (R² 0.728) | 1.145 (R² 0.728) | — |
| MLP | 1.311 (R² 0.593) | 0.924 (R² 0.823) | +42% |

Three things fall out of this table.

**The random split inflates scores by roughly half an RMSE unit.** If you
read a solubility paper reporting ~0.6 RMSE on ESOL without saying how it
split, that is very likely this artifact rather than a better model.

**XGBoost beats the neural network, decisively.** 0.900 against 1.311 on
the honest split — which is the Ch 3 versus Ch 4 comparison the book sets
up but does not close. At ~1100 molecules there is not enough data for the
network to earn its capacity. This is the single most useful thing to
know before spending a week on Ch 4.

**The graph network does not rescue it.** A from-scratch message-passing
GNN (`src/qmprop/gnn.py`, ~80 lines of torch, no torch-geometric) reaches
1.021 — better than the MLP and ridge, still 0.12 behind plain XGBoost on
fingerprints. At 114,721 parameters for 759 training molecules that is
**151 parameters per molecule**, and no amount of architecture fixes that
ratio. Graph networks start winning around 10⁵ molecules; below that the
inductive bias does not pay for the capacity. Running it was still worth
it — a null result you measured is worth more than one you assumed.

A footnote on the ridge row: with a fixed `alpha=1.0` it scored RMSE 2.141
and **R² −0.087** — worse than predicting the mean. At 2247 features on
893 samples a fixed penalty is badly under-regularized. `models.py` uses
`RidgeCV`/`LassoCV`/`ElasticNetCV` so the linear arm is a real baseline
rather than a strawman.

---

## How good could any model get? (Ch 2)

`make chemspace` joins ESOL against **AqSolDB** (9,982 compounds, nine
merged sources) on InChIKey and asks what two independent curations of
the same property say about the same molecule.

| subset | n | RMSE | MAE | max |
|---|---:|---:|---:|---:|
| all shared molecules | 1117 | 0.262 | 0.087 | 2.97 |
| AqSolDB had >1 source | 668 | **0.339** | 0.144 | 2.97 |

ESOL is *one of* AqSolDB's nine sources, so both rows are partly
self-comparison and 0.34 is a **lower bound** on experimental noise, not
an estimate of it. Even so it is the number that matters: the best model
here sits at 0.90 RMSE, comfortably above the floor, so there is real
signal left to capture rather than curation error to overfit. The largest
single disagreement is octadecanol, where the two sources differ by
**2.97 log units** — a factor of ~900 in concentration.

The same script measures how far the test set sits from training:

| split | mean nearest Tanimoto | % below 0.4 |
|---|---:|---:|
| scaffold | 0.393 | **57%** |
| random | 0.546 | 21% |

That gap *is* the +51% RMSE inflation, made concrete. And it is worth
sitting with the second column: 0.4 is the threshold the deployed app
uses to warn that a prediction is out of domain, so **57% of the honest
test set consists of molecules the app itself would flag**. The headline
RMSE is largely measured on exactly those cases — which is the point of
the honest split, not a flaw in it.

---

## Ch 5: the theory layer, and where it breaks

`make theory` solves the two solvable problems three ways. A
finite-difference solver that knows no chemistry (build `H = T + V`,
call `eigh`) reproduces the closed-form energies to **5×10⁻⁶ relative
error** for the particle in a box and **3×10⁻⁵** for the harmonic
oscillator. That agreement is what licenses trusting the same two steps
in Ch 6, where PySCF does them in a Gaussian basis instead of on a grid.

Then the free-electron model predicts polyene π→π* gaps from one number,
the C–C bond length:

| chain | FEM | experiment | error |
|---|---:|---:|---:|
| butadiene (C₄) | 6.00 | 5.92 | **+0.08** |
| hexatriene (C₆) | 3.73 | 4.93 | −1.20 |
| octatetraene (C₈) | 2.70 | 4.41 | −1.71 |
| decapentaene (C₁₀) | 2.11 | 4.02 | −1.91 |

Butadiene looks like a triumph, which is exactly why textbooks stop
there. Extend the series and the model falls apart, and worse, it sends
the gap to **zero** as the chain grows (0.24 eV at 80 carbons) while real
polyenes converge to roughly 2 eV. The missing physics is bond-length
alternation — a real chain is a corrugated box, not a flat one, and the
corrugation holds a gap open at any length.

The lesson travels: **a model agreeing with one data point is not
evidence, it is a coincidence you have not tested yet.** `test_theory.py`
pins the breakdown so nobody quietly "fixes" it into silence.

---

## Ch 7: predictions that show their work

```bash
make explain Q='caffeine'
```

```
Resolved caffeine via pubchem to Cn1c(=O)c2c(ncn2C)n(C)c1=O
### Predicted log S: -0.91 log(mol/L)  (~24,038 mg/L)
This molecule is in the training set; measured -0.88 — a memory check.

| similarity | measured log S | SMILES                             |
|       1.00 |          -0.88 | Cn1c(=O)c2c(ncn2C)n(C)c1=O         |
|       0.53 |          -2.52 | Cn1cnc2c1c(=O)[nH]c(=O)n2C         |

What moved this prediction (exact TreeSHAP, baseline -2.86):
| octanol/water partition (lipophilicity) | -1.03 | +1.83 |
| molecular complexity                    |   617 | -0.40 |
```

Four properties, in order of how much they matter:

1. **The LLM is optional and last.** `render_text()` produces the entire
   explanation deterministically from computed numbers. `render_llm()`
   only rephrases them, and only if `ANTHROPIC_API_KEY` is set. The
   language model never originates a quantity — that is the difference
   between an explanation and a plausible-sounding one.
2. **The attribution is arithmetic, not narrative.** XGBoost's
   `pred_contribs` is exact TreeSHAP; contributions plus bias reproduce
   `model.predict` to ~10⁻⁶ (asserted in `test_explain.py`). A random
   forest gets *no* attribution rather than a substituted global
   importance, which answers a different question.
3. **The table says it is a summary.** A typical prediction has ~880
   non-zero contributions; showing six and implying they sum to the
   answer would be a lie of omission, so the output states the coverage.
4. **Out-of-domain is loud.** `table salt` resolves to `[Cl-].[Na+]`,
   scores 0.10 Tanimoto against training, and is flagged — an ionic solid
   is exactly the chemistry this model has never seen.

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

Two more decisions carry as much weight as that one.

**Scaffold k-fold, not a single split.** An 80/20 split of a
200-molecule subset leaves ~40 test molecules, which cannot resolve the
effect being looked for — the experiment would have been underpowered by
construction, and a null result from it would have meant nothing.
`scaffold_kfold` rotates every molecule through the test set once, so
the comparison runs on 200 out-of-fold predictions instead of 40.
Scaffold groups stay whole, so each fold is still an honest test, and
the extra cost is four more model fits per arm — seconds. The QM cost,
which is the expensive part, does not change at all.

**A paired significance test.** An earlier version quoted
`1.96·σ_y/√n` as the noise floor. That is the confidence interval for
the *mean of y* — a different quantity, and far too wide here: two
models scored on the same molecules make correlated errors, so the
interval on their *difference* is much narrower. Using it would have
buried a real effect. The delta is now bootstrapped by resampling
molecules and recomputing both arms' RMSE on each resample, and the
script reports the CI width so a null result comes with the size of the
effect it could not have detected.

### The answer: no, QM does not help here

Run on **195 molecules** (200 attempted; 5 iodine compounds have no
6-31G\* basis), 5-fold scaffold CV, out-of-fold predictions for every
molecule, 10,000 paired bootstrap resamples:

| model | fingerprints+desc | +QM | delta | 95% CI | verdict |
|---|---:|---:|---:|---:|---|
| Ridge | 0.860 | 0.858 | −0.002 | [−0.011, +0.008] | no measurable effect |
| Random forest | 1.007 | 1.007 | −0.000 | [−0.007, +0.007] | no measurable effect |
| XGBoost | 0.909 | 0.946 | +0.037 | [+0.009, +0.065] | QM *hurts* |
| MLP | 1.352 | 1.170 | −0.182 | [−0.301, −0.060] | QM helps? |

Three of four models show nothing or slightly worse. The MLP appears to
improve — and it is the only interesting number in the table, so it is
the one that had to be checked rather than reported.

**It does not survive its controls.** Rerun the MLP with the QM block
replaced by things that contain no quantum chemistry at all:

| extra block | MLP RMSE | change |
|---|---:|---:|
| none | 1.352 | — |
| real QM | 1.170 | **−0.18** |
| QM rows shuffled (same distributions, no link to the molecule) | 1.228 | −0.12 |
| eight Gaussian noise columns | 1.239 | −0.11 |
| eight copies of heavy-atom count | 1.181 | **−0.17** |

Pure noise reproduces two-thirds of the gain, and molecular size alone
reproduces nearly all of it. The MLP was not learning chemistry; it was
benefiting from eight more well-scaled continuous columns, plus size —
which the QM block is partly a proxy for. The HOMO–LUMO gap correlates
**−0.54** with heavy-atom count here, HOMO **+0.42**, max Mulliken
charge **+0.47**.

So the answer to the question the project was built to ask:

> **B3LYP/6-31G\* orbital energies, dipole moment and Mulliken charges
> add nothing measurable to aqueous solubility prediction once Morgan
> fingerprints and RDKit descriptors are already in hand.**

That is a real finding, not a failed experiment. Solubility is dominated
by polarity and hydrogen bonding, and TPSA, LogP and the H-bond donor
and acceptor counts already encode them — cheaply, and without an hour
of DFT. The honest caveat: at n=195 the mean 95% CI width on the delta
is **0.083 log units**, so an effect smaller than about ±0.04 could not
have been seen here. Absence of evidence at this sample size, not proof
of absence.

Two side observations worth keeping:

- **QM features alone are not useless** — 8 of them reach RMSE 1.66
  (R² 0.27) with a random forest. They carry real signal. It is just
  signal the cheap descriptors already have.
- **Fingerprints alone collapse on this subset** (RMSE 1.93, R² 0.01)
  while descriptors alone nearly match the full set (0.874 vs 0.860).
  With 156 training molecules, 2048 sparse binary columns cannot learn;
  185 dense physicochemical ones can. A small-data result, and a good
  argument for descriptors over fingerprints when data is scarce.

`05_ablation.py` runs those controls automatically whenever any model
reports "QM helps", so this check is part of the pipeline rather than
something I happened to think of once.

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

### Why the subset takes ~45 minutes and not 6 hours

The run uses density fitting (RI-J), which factors the four-index
Coulomb integrals through an auxiliary basis. Measured on a
17-heavy-atom sulfone, one process:

| | time | energy (Ha) | HOMO | gap | dipole |
|---|---:|---:|---:|---:|---:|
| exact | 251.3 s | −1122.559953 | −5.7603 | 5.0784 | 8.9588 |
| RI-J | **47.8 s** | −1122.559930 | −5.7602 | 5.0791 | 8.9587 |

**5.3× for 2.3×10⁻⁵ hartree.** The gap moves 0.0007 eV — four orders of
magnitude below the spread of these features across the dataset, so
nothing downstream can tell the difference. Set `qm.density_fit: false`
if you ever need absolute energies rather than descriptors.

Two smaller notes from the same measurements. `qm.max_memory_mb`
matters more than it looks, because parallel workers multiply it: three
workers swapping an 8 GB machine ran at ~8% of a core each while
`kernel_task` burned a full core on memory compression. And the answers
are identical at every memory budget (gap 5.0784, dipole 8.9588 to four
decimals), so it is a memory knob, not an accuracy one.

## Honest limitations

- **Single points on MMFF geometries**, not DFT-optimized structures.
  ~50× cheaper, fine for descriptors, not publishable as energetics.
- **RI-J, not exact Coulomb integrals.** Verified negligible for these
  descriptors (table above); would need rechecking for energetics.
- **No iodine.** 6-31G\* has no basis functions for I, so those
  molecules come back `ok=False` with a reason and drop out of the
  subset. A documented gap in chemical coverage, not a silent one —
  switch to def2-SVP if it matters for your target.
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

**A bigger dataset.** AqSolDB (~10k molecules) is already downloaded by
`make chemspace`; retargeting the pipeline at it is a config change plus
a column rename, and it is the change most likely to move the GNN result.

**An agent layer.** `explain.py` gives an LLM the facts; the next step is
giving it `qm_descriptors()` and `predict()` as *tools* so it can decide
what to compute. Keep the current invariant — validate every SMILES it
produces through `Chem.MolFromSmiles()`, and never let it originate a
number.

**Better QM.** Geometry optimization instead of single points; a
conformer ensemble; ωB97X-D for dispersion.

---

## Layout

```
config.yaml           every knob, one file
src/qmprop/
  data.py             download, canonicalize, dedupe by InChIKey
  external.py         AqSolDB, PubChem, ChEMBL, the enriched union   (Ch 2)
  features.py         descriptors + ECFP, assembled into blocks
  splits.py           scaffold split (the correction that matters)
  theory.py           box + oscillator, analytic and numerical  (Ch 5)
  qm.py               RDKit geometry -> PySCF -> 8 QM features  (Ch 6)
  gnn.py              message-passing network, plain torch      (Ch 4)
  models.py           ridge/lasso/SVR/RF/XGBoost/MLP registry
  explain.py          resolve -> predict -> TreeSHAP -> prose    (Ch 7)
  evaluate.py         metrics, parity plots, text tables
scripts/
  01_fetch_data.py    ESOL download + clean
  02_featurize.py     descriptors + fingerprints
  02b_chemspace.py    t-SNE/UMAP, noise floor, split separation
  03_run_qm.py        B3LYP single points, parallel + resumable
  04_train.py         Ch 3 vs Ch 4 baselines
  05_ablation.py      does QM help?
  06_theory.py        Ch 5 solvers and the polyene breakdown
  07_explain.py       Ch 7 CLI
  08_gnn.py           Ch 4 stretch goal
  09_scale.py         learning curve on 9x the data (tests a README claim)
notebooks/            Colab quickstart
app/app.py            Gradio predictor, name lookup + explanation
tests/                pytest — 102 tests
```

## Data

ESOL / Delaney: 1128 compounds with measured aqueous solubility, via
DeepChem's mirror. Delaney, *J. Chem. Inf. Comput. Sci.* 44 (2004) 1000.
