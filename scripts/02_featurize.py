#!/usr/bin/env python3
"""Precompute descriptor and fingerprint blocks, cache them as .npz."""

import logging

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.features import descriptor_matrix, morgan_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    smiles = df["smiles"].tolist()

    mcfg = cfg["features"]["morgan"]
    Xm, morgan_names = morgan_matrix(
        smiles, radius=mcfg["radius"], n_bits=mcfg["n_bits"]
    )
    Xd, desc_names = descriptor_matrix(
        smiles, cfg["features"]["descriptors"]["names"]
    )

    out = cfg["data_dir"] / "interim" / "features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        morgan=Xm,
        morgan_names=np.array(morgan_names),
        descriptors=Xd,
        descriptor_names=np.array(desc_names),
        smiles=np.array(smiles),
    )
    print(f"wrote {out}")
    print(f"  morgan      {Xm.shape}")
    print(f"  descriptors {Xd.shape}")


if __name__ == "__main__":
    main()
