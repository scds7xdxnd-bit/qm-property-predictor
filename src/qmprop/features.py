"""Molecular featurization (Ch 1.2 and 1.7).

Two feature families, deliberately kept separate so the ablation can
turn each on and off:

    descriptors -- ~210 interpretable physical quantities (MW, LogP, TPSA)
    morgan      -- a 2048-bit ECFP4 hash of local substructures

Descriptors say *what the molecule is like*; fingerprints say *what
substructures it contains*. They are not redundant, which is why the
combination usually beats either alone.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

log = logging.getLogger(__name__)

# RDKit renamed the fingerprint API in 2023.09; support both.
try:
    from rdkit.Chem import rdFingerprintGenerator

    def _morgan_bits(mol, radius: int, n_bits: int) -> np.ndarray:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        return np.array(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)

except (ImportError, AttributeError):  # pragma: no cover - old RDKit
    from rdkit.Chem import AllChem

    def _morgan_bits(mol, radius: int, n_bits: int) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


def to_mols(smiles: Sequence[str]) -> list:
    """Parse SMILES, raising on the first failure.

    Failing loudly is right here: by this point the data module has
    already dropped everything unparseable, so a None means a bug
    upstream, not a bad input row.
    """
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached featurization: {smi!r}")
        mols.append(mol)
    return mols


def morgan_matrix(
    smiles: Sequence[str], radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, list[str]]:
    """ECFP bit matrix, shape (n_molecules, n_bits)."""
    mols = to_mols(smiles)
    X = np.vstack([_morgan_bits(m, radius, n_bits) for m in mols])
    names = [f"ecfp{2 * radius}_{i}" for i in range(n_bits)]
    return X.astype(np.float32), names


def descriptor_matrix(
    smiles: Sequence[str],
    names: Iterable[str] | None = None,
    drop_degenerate: bool | None = None,
) -> tuple[np.ndarray, list[str]]:
    """RDKit descriptor matrix.

    When selecting from the full catalogue (`names=None`), drops any
    column that is constant or non-finite *in float32*: a handful of
    RDKit descriptors return inf or NaN on exotic structures, and Ipc
    overflows float32 outright on large molecules, either of which will
    silently poison a linear model.

    That filter is TRAINING-TIME logic and must not run at inference.
    Applied to a single molecule every column has zero variance, so it
    would drop all of them and hand the model the wrong feature count.
    So it defaults off whenever explicit `names` are given -- pass the
    exact column list the model was fitted on and you get exactly those
    columns back, in order, with non-finite values zero-filled.
    """
    mols = to_mols(smiles)
    catalogue = dict(Descriptors._descList)
    selected = list(names) if names else list(catalogue)

    unknown = [n for n in selected if n not in catalogue]
    if unknown:
        raise KeyError(f"unknown descriptors: {unknown[:5]}")

    rows = []
    for mol in mols:
        row = []
        for name in selected:
            try:
                row.append(float(catalogue[name](mol)))
            except Exception:  # a few descriptors throw on odd valences
                row.append(np.nan)
        rows.append(row)

    X = np.asarray(rows, dtype=np.float64)

    if drop_degenerate is None:
        drop_degenerate = names is None

    # Cast BEFORE testing finiteness, not after. A value can be perfectly
    # finite in float64 and still overflow float32: RDKit's Ipc reaches
    # 2.8e54 on a ~50-atom molecule, sixteen orders of magnitude past
    # float32's 3.4e38 ceiling. Checking first and casting second lets
    # that column pass the filter and arrive as inf, which surfaces much
    # later as "Input X contains infinity" from inside sklearn. ESOL is
    # small enough never to trigger it; AqSolDB is not.
    with np.errstate(over="ignore"):
        X32 = X.astype(np.float32)

    if not drop_degenerate:
        return np.nan_to_num(X32, nan=0.0, posinf=0.0, neginf=0.0), list(selected)

    finite = np.isfinite(X32).all(axis=0)
    varying = np.nanstd(X, axis=0) > 0
    keep = finite & varying
    dropped = len(selected) - int(keep.sum())
    if dropped:
        log.info("dropped %d constant/non-finite descriptor columns", dropped)

    kept_names = [n for n, k in zip(selected, keep) if k]
    return X32[:, keep], kept_names


def build_features(
    smiles: Sequence[str],
    use_morgan: bool = True,
    use_descriptors: bool = True,
    qm_matrix: np.ndarray | None = None,
    qm_names: Sequence[str] | None = None,
    morgan_kwargs: dict | None = None,
    descriptor_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble one feature block from the requested families.

    `qm_matrix` is passed in rather than computed here: QM features are
    slow enough that they are always precomputed and cached to disk.
    """
    blocks: list[np.ndarray] = []
    names: list[str] = []

    if use_morgan:
        Xm, nm = morgan_matrix(smiles, **(morgan_kwargs or {}))
        blocks.append(Xm)
        names += nm

    if use_descriptors:
        Xd, nd = descriptor_matrix(smiles, descriptor_names)
        blocks.append(Xd)
        names += nd

    if qm_matrix is not None:
        if len(qm_matrix) != len(smiles):
            raise ValueError(
                f"qm_matrix has {len(qm_matrix)} rows, expected {len(smiles)}"
            )
        blocks.append(np.asarray(qm_matrix, dtype=np.float32))
        names += list(qm_names or [f"qm_{i}" for i in range(qm_matrix.shape[1])])

    if not blocks:
        raise ValueError("no feature family selected")

    return np.hstack(blocks), names
