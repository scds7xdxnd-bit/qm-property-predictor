#!/usr/bin/env python3
"""Ch 5: solve the two solvable problems, then watch a model break.

Three things happen here:

  1. The finite-difference solver is checked against closed form for the
     particle in a box and the harmonic oscillator. If this disagrees,
     nothing downstream is trustworthy.
  2. The polyene series is run through the free-electron model and
     scored against measured spectra -- including the chains where it
     fails, which is the point.
  3. Optionally (--dft) the same gaps are computed at B3LYP/6-31G* so
     the crude model and the expensive one sit in one table.

    python scripts/06_theory.py
    python scripts/06_theory.py --dft      # adds PySCF, a few minutes
"""

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.theory import (
    POLYENE_EXPERIMENT_EV,
    POLYENE_SMILES,
    box_energy,
    box_potential,
    box_wavefunction,
    oscillator_energy,
    oscillator_potential,
    oscillator_wavefunction,
    match_sign,
    polyene_gap_ev,
    solve_1d,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("theory")

BOX_LENGTH = 5.0      # bohr
OMEGA = 1.0           # hartree


def validate_solver() -> dict:
    """Numerical vs analytic for both potentials. Returns worst errors."""
    n_box = solve_1d(box_potential(BOX_LENGTH), 0.0, BOX_LENGTH,
                     n_grid=2000, n_states=5)
    e_box = box_energy(np.arange(1, 6), BOX_LENGTH)

    n_osc = solve_1d(oscillator_potential(OMEGA), -10.0, 10.0,
                     n_grid=2000, n_states=5)
    e_osc = oscillator_energy(np.arange(5), OMEGA)

    box_err = float(np.max(np.abs(n_box.energies - e_box) / e_box))
    osc_err = float(np.max(np.abs(n_osc.energies - e_osc) / e_osc))

    print("\n== finite-difference solver vs closed form ==\n")
    print(f"{'state':>6} {'box exact':>11} {'box numeric':>12} "
          f"{'osc exact':>11} {'osc numeric':>12}")
    for i in range(5):
        print(f"{i:>6} {e_box[i]:11.6f} {n_box.energies[i]:12.6f} "
              f"{e_osc[i]:11.6f} {n_osc.energies[i]:12.6f}")
    print(f"\nworst relative error: box {box_err:.2e}, oscillator {osc_err:.2e}")

    return {"box_max_rel_error": box_err, "oscillator_max_rel_error": osc_err,
            "box": n_box, "oscillator": n_osc}


def polyene_table(use_dft: bool) -> list[dict]:
    """Free-electron model against experiment, optionally against B3LYP."""
    dft_gaps: dict[int, float] = {}
    if use_dft:
        from qmprop.qm import qm_descriptors
        cfg = load_config()["qm"]
        for n, smi in POLYENE_SMILES.items():
            res = qm_descriptors(smi, basis=cfg["basis"], xc=cfg["xc"],
                                 max_heavy_atoms=cfg["max_heavy_atoms"],
                                 conformer_seed=cfg["conformer_seed"],
                                 mmff_max_iters=cfg["mmff_max_iters"])
            if res.ok:
                dft_gaps[n] = res.qm_gap_ev
                log.info("B3LYP %2d C: gap %.2f eV", n, res.qm_gap_ev)
            else:
                log.warning("B3LYP %2d C failed: %s", n, res.reason)

    rows = []
    for n in sorted(POLYENE_EXPERIMENT_EV):
        fem = polyene_gap_ev(n)
        expt = POLYENE_EXPERIMENT_EV[n]
        row = {"n_carbons": n, "smiles": POLYENE_SMILES[n],
               "fem_ev": fem, "experiment_ev": expt, "fem_error_ev": fem - expt}
        if n in dft_gaps:
            row["b3lyp_gap_ev"] = dft_gaps[n]
            row["b3lyp_error_ev"] = dft_gaps[n] - expt
        rows.append(row)

    print("\n== polyene pi -> pi* gaps ==\n")
    header = f"{'n_C':>4} {'SMILES':<22} {'FEM':>7} {'expt':>7} {'err':>7}"
    if dft_gaps:
        header += f" {'B3LYP':>7} {'err':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (f"{r['n_carbons']:>4} {r['smiles']:<22} {r['fem_ev']:7.2f} "
                f"{r['experiment_ev']:7.2f} {r['fem_error_ev']:+7.2f}")
        if "b3lyp_gap_ev" in r:
            line += f" {r['b3lyp_gap_ev']:7.2f} {r['b3lyp_error_ev']:+7.2f}"
        print(line)

    print("\nThe free-electron model reproduces butadiene to "
          f"{abs(rows[0]['fem_error_ev']):.2f} eV and then degrades "
          f"monotonically to {rows[-1]['fem_error_ev']:+.2f} eV.")
    print("It is missing bond-length alternation: a real polyene is a "
          "corrugated box, not a flat one, so its gap stays open")
    print(f"as the chain grows. FEM instead sends it to zero -- "
          f"{polyene_gap_ev(80):.2f} eV at 80 carbons, which is nonsense.")
    if dft_gaps:
        print("\nB3LYP Kohn-Sham gaps are NOT excitation energies; they are "
              "listed to show scale, not to be scored as spectroscopy.")
    return rows


def make_figure(solved: dict, rows: list[dict], path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    x = solved["box"].x
    for i in range(3):
        ax.plot(x, solved["box"].wavefunctions[i], lw=2,
                label=f"n={i + 1} numeric")
        ax.plot(x, match_sign(solved["box"].wavefunctions[i],
                               box_wavefunction(i + 1, BOX_LENGTH, x)),
                "k--", lw=1, alpha=.7)
    ax.set_title("particle in a box\n(dashed = analytic, sign-aligned)")
    ax.set_xlabel("x (bohr)")
    ax.set_ylabel(r"$\psi$")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = solved["oscillator"].x
    keep = (x > -6) & (x < 6)
    for i in range(3):
        ax.plot(x[keep], solved["oscillator"].wavefunctions[i][keep], lw=2,
                label=f"n={i} numeric")
        ax.plot(x[keep], match_sign(solved["oscillator"].wavefunctions[i],
                                    oscillator_wavefunction(i, OMEGA, x))[keep],
                "k--", lw=1, alpha=.7)
    ax.plot(x[keep], 0.5 * OMEGA**2 * x[keep] ** 2 / 10, color="grey",
            lw=1, alpha=.5, label="V(x)/10")
    ax.set_title("harmonic oscillator\n(dashed = analytic, sign-aligned)")
    ax.set_xlabel("x (bohr)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ns = [r["n_carbons"] for r in rows]
    ax.plot(ns, [r["experiment_ev"] for r in rows], "o-", label="experiment")
    ax.plot(ns, [r["fem_ev"] for r in rows], "s--", label="free-electron model")
    if "b3lyp_gap_ev" in rows[0]:
        ax.plot(ns, [r.get("b3lyp_gap_ev", np.nan) for r in rows], "^:",
                label="B3LYP/6-31G* KS gap")
    ax.set_title("where the box model breaks")
    ax.set_xlabel("carbons in conjugated chain")
    ax.set_ylabel("gap (eV)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dft", action="store_true",
                    help="also run B3LYP/6-31G* on the polyenes (slow)")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["output_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    solved = validate_solver()
    rows = polyene_table(args.dft)
    make_figure(solved, rows, out / "figures" / "theory.png")

    with open(out / "theory.json", "w", encoding="utf-8") as fh:
        json.dump({
            "solver_validation": {
                "box_max_rel_error": solved["box_max_rel_error"],
                "oscillator_max_rel_error": solved["oscillator_max_rel_error"],
            },
            "polyenes": rows,
        }, fh, indent=2)
    print(f"wrote {out / 'theory.json'}")


if __name__ == "__main__":
    main()
