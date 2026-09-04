"""Analytic and numerical solutions of the two solvable problems (Ch 5).

Ch 6 hands you HOMO and LUMO from a DFT black box. This module exists so
that those numbers are not magic. Everything here is written from the
Schrodinger equation with numpy and nothing else.

Three layers, each checking the one before it:

  1. `box_*` and `oscillator_*` -- closed-form energies and wavefunctions
     for the particle in a box and the harmonic oscillator.
  2. `solve_1d` -- a finite-difference solver that knows no chemistry at
     all: hand it any V(x) and it builds H = T + V and diagonalizes.
     Run it on the two potentials above and it reproduces layer 1 to
     four digits. That agreement is the whole point -- it is how you
     trust the solver on a potential with no closed form.
  3. `polyene_gap_ev` -- the free-electron model: treat the pi electrons
     of a conjugated chain as particles in a 1-D box and predict the
     HOMO-LUMO gap from one number, the C-C bond length.

Layer 3 is the bridge to Ch 6, and it is worth running precisely
because it *fails*. It reproduces butadiene to within 0.08 eV, which
looks like a triumph -- and is why textbooks stop there. Extend the
series and the error grows monotonically to -1.9 eV by decapentaene,
because the model predicts the gap collapsing as 1/N toward zero while
real polyenes converge to roughly 2 eV. The physics it is missing is
bond-length alternation: a real chain is not a flat-bottomed box but a
corrugated one, and that corrugation holds a gap open at any length.

That is the honest lesson to carry into Ch 6. A model agreeing with one
data point is not evidence; it is a coincidence you have not tested yet.
`scripts/06_theory.py` runs the whole series so the breakdown is visible
rather than assumed.

Atomic units throughout (hbar = m_e = e = 1); lengths in bohr, energies
in hartree. The public helpers convert to eV and angstrom at the edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Length of one conjugated C-C bond, the single empirical input to the
# free-electron model. 1.40 A is the usual compromise between the single
# (1.54) and double (1.34) bonds it is averaging over.
CONJUGATED_BOND_ANGSTROM = 1.40


# --------------------------------------------------------------------------
# Layer 1: closed form
# --------------------------------------------------------------------------

def box_energy(n: int | np.ndarray, length: float) -> np.ndarray:
    """E_n for a particle in a box with infinite walls, in hartree.

    E_n = n^2 * pi^2 / (2 L^2)   (atomic units, m = 1)

    The n^2 is the fact worth carrying around: levels spread out as you
    climb, so a longer box (bigger L) does not just lower every level,
    it *compresses* them. That is why long conjugated chains absorb red
    and short ones absorb UV.
    """
    n = np.asarray(n, dtype=float)
    if np.any(n < 1):
        raise ValueError("quantum number n starts at 1 for a box")
    if length <= 0:
        raise ValueError("box length must be positive")
    return n**2 * math.pi**2 / (2.0 * length**2)


def box_wavefunction(n: int, length: float, x: np.ndarray) -> np.ndarray:
    """psi_n(x) = sqrt(2/L) sin(n pi x / L), zero outside the walls."""
    x = np.asarray(x, dtype=float)
    psi = np.sqrt(2.0 / length) * np.sin(n * math.pi * x / length)
    return np.where((x < 0) | (x > length), 0.0, psi)


def oscillator_energy(n: int | np.ndarray, omega: float) -> np.ndarray:
    """E_n = omega (n + 1/2), in hartree. n starts at 0.

    Note the contrast with the box: levels are evenly spaced, and the
    ground state sits at omega/2 rather than 0. That residual is zero-
    point energy -- it is why a bond never stops vibrating, and why
    comparing two calculated energies without ZPE corrections is wrong.
    """
    n = np.asarray(n, dtype=float)
    if np.any(n < 0):
        raise ValueError("quantum number n starts at 0 for an oscillator")
    if omega <= 0:
        raise ValueError("omega must be positive")
    return omega * (n + 0.5)


def oscillator_wavefunction(n: int, omega: float, x: np.ndarray) -> np.ndarray:
    """psi_n(x) = N_n H_n(sqrt(omega) x) exp(-omega x^2 / 2).

    The normalization carries 2^n n!, which overflows float64 around
    n = 150, so it is built in log space.
    """
    x = np.asarray(x, dtype=float)
    log_norm = (
        0.25 * math.log(omega / math.pi)
        - 0.5 * (n * math.log(2.0) + math.lgamma(n + 1))
    )
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    hermite = np.polynomial.hermite.hermval(np.sqrt(omega) * x, coeffs)
    return math.exp(log_norm) * hermite * np.exp(-omega * x**2 / 2.0)


# --------------------------------------------------------------------------
# Layer 2: the general solver
# --------------------------------------------------------------------------

@dataclass
class Spectrum:
    """Eigenvalues (hartree) and eigenvectors on the grid they were found on."""

    energies: np.ndarray      # (k,)
    wavefunctions: np.ndarray  # (k, n_grid), normalized so sum |psi|^2 dx = 1
    x: np.ndarray             # (n_grid,)

    @property
    def gap(self) -> float:
        """Spacing of the lowest two levels, in hartree."""
        return float(self.energies[1] - self.energies[0])


def solve_1d(
    potential: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    n_grid: int = 2000,
    n_states: int = 6,
) -> Spectrum:
    """Diagonalize H = -1/2 d2/dx2 + V(x) on a finite-difference grid.

    The kinetic operator becomes a tridiagonal matrix by way of the
    three-point second difference,

        psi''(x) ~ [psi(x-h) - 2 psi(x) + psi(x+h)] / h^2

    and the potential is just a diagonal. Then `eigh` does the rest.
    Dirichlet walls are implicit: the grid excludes its own endpoints,
    so psi is pinned to zero there. Put `x_min`/`x_max` far enough out
    that the states you care about have already decayed, or you are
    measuring your box instead of your potential.

    Cost is O(n_grid^3) -- 2000 points is about a second. This is the
    honest, stupid way to solve a 1-D problem, and it is the same two
    steps (build a matrix in some representation, diagonalize it) that
    PySCF performs in Ch 6 with a Gaussian basis instead of a grid.
    """
    if n_grid < n_states + 2:
        raise ValueError("grid is too coarse to hold that many states")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")

    # Interior points only -- endpoints are the walls.
    h = (x_max - x_min) / (n_grid + 1)
    x = x_min + h * np.arange(1, n_grid + 1)

    main = 1.0 / h**2 + potential(x)
    off = -0.5 / h**2 * np.ones(n_grid - 1)
    hamiltonian = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    energies, vectors = np.linalg.eigh(hamiltonian)

    psi = vectors[:, :n_states].T
    psi = psi / np.sqrt(np.sum(psi**2, axis=1, keepdims=True) * h)
    psi = np.array([_canonical_sign(p) for p in psi])

    return Spectrum(energies=energies[:n_states], wavefunctions=psi, x=x)


def _canonical_sign(psi: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Pin the arbitrary global sign of an eigenvector, deterministically.

    `eigh` returns psi up to a factor of -1; only |psi|^2 is observable.
    Leaving it unpinned makes plots and regression tests flap. The
    obvious convention -- "make the biggest lobe positive" -- is *not*
    well defined for a symmetric state like the n=2 box level, whose two
    lobes are equal to within rounding, so the winner is decided by
    floating-point noise. Anchoring on the first lobe that clears the
    tolerance is unambiguous for every state.
    """
    significant = np.flatnonzero(np.abs(psi) > tol * np.max(np.abs(psi)))
    if significant.size and psi[significant[0]] < 0:
        return -psi
    return psi


def match_sign(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Flip `target` to the sign convention of `reference`.

    For comparing a numerical state against a closed-form one: they are
    the same physical state even when they differ by an overall -1, so
    the comparison has to be sign-insensitive to mean anything.
    """
    overlap = float(np.dot(reference, target))
    return -target if overlap < 0 else target


def box_potential(length: float) -> Callable[[np.ndarray], np.ndarray]:
    """V = 0 inside, and the walls are supplied by the grid boundary."""
    return lambda x: np.zeros_like(x)


def oscillator_potential(omega: float) -> Callable[[np.ndarray], np.ndarray]:
    """V = 1/2 omega^2 x^2 (atomic units, m = 1)."""
    return lambda x: 0.5 * omega**2 * x**2


# --------------------------------------------------------------------------
# Layer 3: the free-electron model, and the bridge to Ch 6
# --------------------------------------------------------------------------

def polyene_box_length(n_carbons: int, bond_angstrom: float = CONJUGATED_BOND_ANGSTROM) -> float:
    """Box length in bohr for a linear conjugated chain of `n_carbons`.

    The chain has n-1 bonds; the convention is to let the electron cloud
    spill half a bond past each terminal carbon, giving (n-1) + 1 = n
    bond lengths total. That overhang is a fudge, and it is doing real
    work -- drop it and the predicted gaps come out roughly 30% too wide.
    """
    if n_carbons < 2:
        raise ValueError("need at least two carbons to have a conjugated chain")
    return n_carbons * bond_angstrom * ANGSTROM_TO_BOHR


def polyene_gap_ev(
    n_carbons: int,
    bond_angstrom: float = CONJUGATED_BOND_ANGSTROM,
) -> float:
    """HOMO-LUMO gap in eV for a linear polyene, free-electron model.

    Each carbon in the conjugated chain contributes one pi electron.
    Two electrons per level, so the HOMO is level n_carbons/2 and the
    LUMO the one above:

        dE = (n_LUMO^2 - n_HOMO^2) * pi^2 / (2 L^2)

    Only even chains are physical here (a full pi shell); odd counts are
    radicals and this model has nothing to say about them.

    Accurate for butadiene and increasingly wrong after it -- see the
    module docstring. Do not read a small `n_carbons` agreement as
    validation of the model.
    """
    if n_carbons % 2:
        raise ValueError("free-electron model assumes a closed pi shell (even carbons)")
    length = polyene_box_length(n_carbons, bond_angstrom)
    n_homo = n_carbons // 2
    n_lumo = n_homo + 1
    gap_hartree = box_energy(n_lumo, length) - box_energy(n_homo, length)
    return float(gap_hartree * HARTREE_TO_EV)


# Measured lowest strong pi -> pi* absorptions, gas phase, for the linear
# all-trans polyenes (approximate vertical excitations, +/- ~0.1 eV).
# Used to score the model rather than admire it.
POLYENE_EXPERIMENT_EV = {
    4: 5.92,    # 1,3-butadiene
    6: 4.93,    # 1,3,5-hexatriene
    8: 4.41,    # 1,3,5,7-octatetraene
    10: 4.02,   # decapentaene
}

POLYENE_SMILES = {
    4: "C=CC=C",
    6: "C=CC=CC=C",
    8: "C=CC=CC=CC=C",
    10: "C=CC=CC=CC=CC=C",
}
