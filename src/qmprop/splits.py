"""Scaffold splitting (the correction the book omits).

A random split scatters near-identical analogs across train and test, so
the model is graded on molecules it has effectively already seen. Scores
come out optimistic by a wide margin -- often 0.2-0.4 RMSE on ESOL.

Splitting by Bemis-Murcko scaffold instead puts every member of a
chemical series on one side of the wall. It is harder, and it is the
number that survives contact with a new compound.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """The molecule's ring-system core, as SMILES.

    Acyclic molecules have no Murcko scaffold and return ''. They are all
    grouped together, which is the conventional (if blunt) treatment.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )


def scaffold_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.0,
    frac_test: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices by scaffold, largest scaffold groups first.

    Deterministic given the same input order; `seed` only shuffles groups
    of equal size so ties do not always break the same way.

    Returns (train_idx, valid_idx, test_idx) as integer arrays.
    """
    total = frac_train + frac_valid + frac_test
    if not np.isclose(total, 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))

    n = len(smiles)
    n_train = int(np.floor(frac_train * n))
    n_valid = int(np.floor(frac_valid * n))

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for group in ordered:
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(valid) + len(group) <= n_valid:
            valid.extend(group)
        else:
            test.extend(group)

    log.info(
        "scaffold split: %d scaffolds -> train %d / valid %d / test %d",
        len(groups), len(train), len(valid), len(test),
    )
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def random_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.0,
    frac_test: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split -- included only so the ablation can quantify the gap."""
    n = len(smiles)
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(np.floor(frac_train * n))
    n_valid = int(np.floor(frac_valid * n))
    return (
        np.sort(idx[:n_train]),
        np.sort(idx[n_train:n_train + n_valid]),
        np.sort(idx[n_train + n_valid:]),
    )


SPLITTERS = {"scaffold": scaffold_split, "random": random_split}


def get_split(method: str):
    if method not in SPLITTERS:
        raise ValueError(f"unknown split method {method!r}; use {list(SPLITTERS)}")
    return SPLITTERS[method]
