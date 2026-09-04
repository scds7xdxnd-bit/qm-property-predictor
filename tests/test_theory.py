"""The theory layer is testable in a way the ML layer is not: the answers
are known in closed form, so the numerical solver can be held to them."""

import math

import numpy as np
import pytest

from qmprop.theory import (
    HARTREE_TO_EV,
    POLYENE_EXPERIMENT_EV,
    box_energy,
    box_potential,
    box_wavefunction,
    oscillator_energy,
    oscillator_potential,
    match_sign,
    oscillator_wavefunction,
    polyene_gap_ev,
    solve_1d,
)


# --- layer 1 against pencil and paper -------------------------------------

def test_box_energy_scales_as_n_squared():
    e = box_energy([1, 2, 3], 1.0)
    assert e[1] / e[0] == pytest.approx(4.0)
    assert e[2] / e[0] == pytest.approx(9.0)


def test_box_energy_falls_as_one_over_l_squared():
    assert box_energy(1, 2.0) == pytest.approx(box_energy(1, 1.0) / 4.0)


def test_oscillator_levels_are_evenly_spaced():
    e = oscillator_energy(np.arange(5), 0.7)
    assert np.allclose(np.diff(e), 0.7)


def test_oscillator_ground_state_is_zero_point_energy():
    assert oscillator_energy(0, 1.3) == pytest.approx(0.65)


@pytest.mark.parametrize("bad", [0, -1])
def test_box_rejects_nonphysical_quantum_number(bad):
    with pytest.raises(ValueError):
        box_energy(bad, 1.0)


# --- wavefunctions are normalized and orthogonal --------------------------

def test_box_wavefunctions_normalized_and_orthogonal():
    x = np.linspace(0, 1, 20001)
    psi1 = box_wavefunction(1, 1.0, x)
    psi2 = box_wavefunction(2, 1.0, x)
    assert np.trapezoid(psi1**2, x) == pytest.approx(1.0, abs=1e-4)
    assert np.trapezoid(psi1 * psi2, x) == pytest.approx(0.0, abs=1e-4)


def test_oscillator_wavefunctions_normalized_and_orthogonal():
    x = np.linspace(-12, 12, 40001)
    psi0 = oscillator_wavefunction(0, 1.0, x)
    psi3 = oscillator_wavefunction(3, 1.0, x)
    assert np.trapezoid(psi0**2, x) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(psi0 * psi3, x) == pytest.approx(0.0, abs=1e-6)


def test_node_count_equals_quantum_number():
    """n nodes for oscillator state n -- the standard sanity check.

    Exact zeros have to be dropped before counting: this grid contains
    x = 0 exactly, np.sign(0) is 0, and a -1 -> 0 -> +1 run would
    otherwise be counted as two crossings instead of one.
    """
    x = np.linspace(-8, 8, 4001)
    for n in range(4):
        psi = oscillator_wavefunction(n, 1.0, x)
        interior = psi[50:-50]           # ignore numerical fuzz in the tails
        signs = np.sign(interior[interior != 0.0])
        nodes = np.sum(np.diff(signs) != 0)
        assert nodes == n


# --- layer 2 must reproduce layer 1 ---------------------------------------

def test_solver_reproduces_box_energies():
    exact = box_energy([1, 2, 3, 4], 5.0)
    found = solve_1d(box_potential(5.0), 0.0, 5.0, n_grid=2000, n_states=4)
    assert np.allclose(found.energies, exact, rtol=1e-4)


def test_solver_reproduces_oscillator_energies():
    exact = oscillator_energy(np.arange(4), 1.0)
    found = solve_1d(oscillator_potential(1.0), -10, 10, n_grid=2000, n_states=4)
    assert np.allclose(found.energies, exact, rtol=1e-4)


def test_solver_converges_with_grid_density():
    """Finer grid, smaller error -- if this fails the discretization is wrong."""
    exact = float(box_energy(1, 4.0))
    errors = [
        abs(solve_1d(box_potential(4.0), 0, 4.0, n_grid=n, n_states=2).energies[0] - exact)
        for n in (50, 200, 800)
    ]
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_solver_wavefunction_matches_analytic_shape(n):
    """n=2 is the one that matters: its lobes are equal, so any
    largest-lobe sign convention decides by rounding noise."""
    found = solve_1d(box_potential(3.0), 0.0, 3.0, n_grid=1500, n_states=3)
    exact = box_wavefunction(n, 3.0, found.x)
    psi = found.wavefunctions[n - 1]
    assert np.max(np.abs(psi - match_sign(psi, exact))) < 1e-3


def test_solver_sign_convention_is_deterministic():
    """Same problem, same signs -- twice, including the symmetric state."""
    a = solve_1d(box_potential(3.0), 0.0, 3.0, n_grid=1500, n_states=4)
    b = solve_1d(box_potential(3.0), 0.0, 3.0, n_grid=1500, n_states=4)
    assert np.array_equal(np.sign(a.wavefunctions), np.sign(b.wavefunctions))


def test_solver_states_start_positive():
    """The stated convention: first significant lobe is positive."""
    found = solve_1d(box_potential(3.0), 0.0, 3.0, n_grid=1500, n_states=4)
    for psi in found.wavefunctions:
        first = psi[np.flatnonzero(np.abs(psi) > 1e-6 * np.max(np.abs(psi)))[0]]
        assert first > 0


def test_solver_rejects_inverted_interval():
    with pytest.raises(ValueError):
        solve_1d(box_potential(1.0), 1.0, 0.0)


# --- layer 3: the model has to actually predict something -----------------

def test_polyene_gap_shrinks_with_chain_length():
    gaps = [polyene_gap_ev(n) for n in (4, 6, 8, 10)]
    assert gaps == sorted(gaps, reverse=True)


def test_polyene_gap_matches_butadiene():
    """The one chain the free-electron model gets right, to 0.1 eV."""
    assert abs(polyene_gap_ev(4) - POLYENE_EXPERIMENT_EV[4]) < 0.15


def test_polyene_error_grows_monotonically_with_chain_length():
    """Pins the model's known breakdown so nobody 'fixes' it into silence.

    Agreement at butadiene is not evidence the model works -- the error
    is a one-way ratchet after it, and always in the same direction.
    """
    errors = [polyene_gap_ev(n) - POLYENE_EXPERIMENT_EV[n]
              for n in sorted(POLYENE_EXPERIMENT_EV)]
    assert all(e < 0 for e in errors[1:]), "model should underestimate"
    assert errors == sorted(errors, reverse=True), "error should worsen"
    assert errors[-1] < -1.5, "decapentaene should be badly wrong"


def test_polyene_gap_collapses_unphysically_for_long_chains():
    """The qualitative failure: FEM sends the gap to 0 as 1/N, but real
    polyenes converge to roughly 2 eV. Documented, not accidental."""
    assert polyene_gap_ev(80) < 0.5
    assert polyene_gap_ev(40) == pytest.approx(2 * polyene_gap_ev(80), rel=0.05)


def test_polyene_rejects_open_shell():
    with pytest.raises(ValueError):
        polyene_gap_ev(5)
