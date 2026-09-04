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


def scaffold_kfold(
    smiles: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Scaffold-grouped k-fold. Yields (train_idx, test_idx) per fold.

    Why this exists: a single 80/20 scaffold split of a 200-molecule QM
    subset leaves 40 test molecules, and 40 is not enough to resolve
    whether eight quantum features moved RMSE. Rotating every molecule
    through the test set once gives out-of-fold predictions for all 200
    -- five times the comparison data for five times a training cost
    that is measured in seconds.

    Scaffolds stay whole. A group is never split across folds, so the
    honesty of the scaffold split is preserved fold by fold: each test
    fold contains chemical series the corresponding training set has
    never seen.

    Folds are balanced greedily, largest group first into whichever fold
    is currently smallest. With a few dominant scaffolds (benzene is 253
    of 1117 molecules here) the folds cannot come out exactly equal, and
    forcing them equal would mean splitting a group -- which is the one
    thing this must not do.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(smiles) < n_folds:
        raise ValueError(f"{len(smiles)} molecules cannot fill {n_folds} folds")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi)].append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    # Shuffle within each size tier so ties do not always break the same way.
    sizes = defaultdict(list)
    for g in ordered:
        sizes[len(g)].append(g)
    ordered = []
    for size in sorted(sizes, reverse=True):
        tier = sizes[size]
        rng.shuffle(tier)
        ordered.extend(tier)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for group in ordered:
        smallest = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[smallest].extend(group)

    empty = [i for i, f in enumerate(folds) if not f]
    if empty:
        raise ValueError(
            f"folds {empty} came out empty -- too few scaffolds "
            f"({len(groups)}) for {n_folds} folds"
        )

    all_idx = np.arange(len(smiles))
    out = []
    for f in folds:
        test = np.sort(np.array(f, dtype=int))
        train = np.sort(np.setdiff1d(all_idx, test))
        out.append((train, test))
    return out


SPLITTERS = {"scaffold": scaffold_split, "random": random_split}


def get_split(method: str):
    if method not in SPLITTERS:
        raise ValueError(f"unknown split method {method!r}; use {list(SPLITTERS)}")
    return SPLITTERS[method]
