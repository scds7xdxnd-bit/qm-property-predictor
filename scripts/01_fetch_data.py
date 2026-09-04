#!/usr/bin/env python3
"""Download ESOL, canonicalize, deduplicate, write data/processed/dataset.csv."""

import logging

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.data import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    cfg = load_config()
    df = load_dataset(cfg)

    out = cfg["data_dir"] / "processed" / "dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    target = cfg["dataset"]["target_name"]
    print(f"\nwrote {out}  ({len(df)} molecules)")
    print(df[target].describe().to_string())


if __name__ == "__main__":
    main()
