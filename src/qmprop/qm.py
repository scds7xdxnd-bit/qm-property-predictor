"""Quantum-chemical descriptors via PySCF (Ch 5 and 6).

The pipeline for one molecule:

    SMILES -> add H -> ETKDG embed -> MMFF optimize   (RDKit, cheap)
           -> B3LYP/6-31G* single point               (PySCF, expensive)
           -> HOMO, LUMO, gap, dipole, Mulliken range

This is the free substitute for the Gaussian workflow in Ch 6.5. The
concepts are identical -- build a geometry, choose a method and basis,
run, parse the orbital energies -- and `pip install pyscf` costs nothing.

Two honest caveats the book's framing can obscure:

  * This is a SINGLE POINT on an MMFF geometry, not a DFT geometry
    optimization. It is roughly 50x cheaper and good enough for
    descriptors. It is not good enough to publish a barrier height.
  * Orbital energies from Kohn-Sham DFT are not ionization potentials.
    They correlate with reactivity, which is all a feature needs to do.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988

QM_FEATURE_NAMES = [
    "qm_homo_ev",
    "qm_lumo_ev",
    "qm_gap_ev",
    "qm_dipole_debye",
    "qm_charge_min",
    "qm_charge_max",
    "qm_charge_span",
    "qm_energy_per_electron_ha",
]


@dataclass
class QMResult:
    smiles: str
    ok: bool
    reason: str = ""
    qm_homo_ev: float = np.nan
    qm_lumo_ev: float = np.nan
    qm_gap_ev: float = np.nan
    qm_dipole_debye: float = np.nan
    qm_charge_min: float = np.nan
    qm_charge_max: float = np.nan
    qm_charge_span: float = np.nan
    qm_energy_per_electron_ha: float = np.nan

    def as_row(self) -> dict:
        return asdict(self)


def embed_3d(smiles: str, seed: int = 0xF00D, max_iters: int = 500):
    """SMILES -> 3D molecule with hydrogens, MMFF-relaxed.

    Returns (mol, reason). `mol` is None when embedding fails, which
    happens for a small fraction of strained cages and large macrocycles.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable SMILES"

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry with random coordinates before giving up.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None, "ETKDG embedding failed"

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception as exc:  # MMFF has no parameters for some elements
        log.debug("MMFF skipped for %s: %s", smiles, exc)

    return mol, ""


def _pyscf_atoms(mol) -> list[tuple[str, tuple[float, float, float]]]:
    """RDKit conformer -> PySCF atom list, coordinates in Angstrom."""
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    return atoms


def qm_descriptors(
    smiles: str,
    basis: str = "6-31G*",
    xc: str = "b3lyp",
    max_heavy_atoms: int = 20,
    conformer_seed: int = 0xF00D,
    mmff_max_iters: int = 500,
    max_memory_mb: int = 900,
) -> QMResult:
    """Compute QM features for one molecule. Never raises -- failures
    come back as `ok=False` with a reason, so a batch run does not die
    on molecule 137 of 200.

    `max_memory_mb` is the one knob worth understanding. PySCF's default
    is 4000 MB, and it will happily store the whole two-electron integral
    tensor if it believes that fits. For 6-31G* on 20 heavy atoms that is
    ~250 basis functions, and 250^4/8 doubles is about 3.9 GB -- so it
    decides it fits, allocates it, and three parallel workers ask an 8 GB
    laptop for 10 GB. Measured here: resident sets of 2.6-3.6 GB per
    worker, kernel_task pinned at ~95% running the memory compressor, and
    each worker getting 8% of a core.

    Setting a small budget makes PySCF fall back to direct SCF, which
    recomputes integrals instead of storing them. More arithmetic, far
    less memory, and on a machine that would otherwise swap it is faster
    by a wide margin. Raise it if you have the RAM.
    """
    mol2d = Chem.MolFromSmiles(smiles)
    if mol2d is None:
        return QMResult(smiles=smiles, ok=False, reason="unparseable SMILES")

    n_heavy = mol2d.GetNumHeavyAtoms()
    if n_heavy > max_heavy_atoms:
        return QMResult(
            smiles=smiles, ok=False,
            reason=f"too large ({n_heavy} > {max_heavy_atoms} heavy atoms)",
        )

    mol3d, reason = embed_3d(smiles, conformer_seed, mmff_max_iters)
    if mol3d is None:
        return QMResult(smiles=smiles, ok=False, reason=reason)

    try:
        from pyscf import dft, gto
    except ImportError:
        return QMResult(smiles=smiles, ok=False, reason="pyscf not installed")

    charge = Chem.GetFormalCharge(mol3d)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol3d.GetAtoms())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pyscf_mol = gto.M(
                atom=_pyscf_atoms(mol3d),
                basis=basis,
                charge=charge,
                spin=n_radical,   # 2S, i.e. count of unpaired electrons
                verbose=0,
                unit="Angstrom",
                max_memory=max_memory_mb,
            )

            mf = dft.UKS(pyscf_mol) if n_radical else dft.RKS(pyscf_mol)
            mf.xc = xc
            mf.max_memory = max_memory_mb
            energy = mf.kernel()

        if not mf.converged:
            return QMResult(smiles=smiles, ok=False, reason="SCF did not converge")

        # Flatten the alpha/beta axis for open-shell (UKS) results so the
        # HOMO/LUMO extraction below is the same code either way.
        mo_energy = np.concatenate([np.ravel(e) for e in np.atleast_2d(mf.mo_energy)])
        mo_occ = np.concatenate([np.ravel(o) for o in np.atleast_2d(mf.mo_occ)])

        occupied = mo_energy[mo_occ > 0]
        virtual = mo_energy[mo_occ == 0]
        if occupied.size == 0 or virtual.size == 0:
            return QMResult(smiles=smiles, ok=False, reason="no frontier orbitals")

        homo = float(occupied.max()) * HARTREE_TO_EV
        lumo = float(virtual.min()) * HARTREE_TO_EV

        dipole = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, charges = mf.mulliken_pop(verbose=0)
        charges = np.asarray(charges, dtype=float)

        return QMResult(
            smiles=smiles,
            ok=True,
            qm_homo_ev=homo,
            qm_lumo_ev=lumo,
            qm_gap_ev=lumo - homo,
            qm_dipole_debye=dipole,
            qm_charge_min=float(charges.min()),
            qm_charge_max=float(charges.max()),
            qm_charge_span=float(charges.max() - charges.min()),
            qm_energy_per_electron_ha=float(energy) / pyscf_mol.nelectron,
        )

    except Exception as exc:
        return QMResult(
            smiles=smiles, ok=False, reason=f"{type(exc).__name__}: {exc}"[:200]
        )
