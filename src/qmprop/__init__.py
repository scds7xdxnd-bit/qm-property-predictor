"""QM-augmented molecular property prediction.

Chapter map (파이썬을 이용한 화학 인공지능):
    data      -> Ch 2   dataset assembly, canonicalization, deduplication
    features  -> Ch 1   descriptors (1.2) and Morgan fingerprints (1.7)
    splits    -> Ch 3   scaffold splitting (the correction the book omits)
    qm        -> Ch 5-6 PySCF single points -> HOMO/LUMO/dipole features
    models    -> Ch 3-4 ridge, random forest, XGBoost, MLP
    evaluate  -> Ch 3.13 metrics and parity plots

`load_config` is resolved lazily (PEP 562) so that importing a leaf
module -- `qmprop.splits`, say -- does not drag in PyYAML. Keeps the
tests runnable on a minimal install.
"""

__version__ = "0.1.0"

__all__ = ["load_config", "__version__"]


def __getattr__(name: str):
    if name == "load_config":
        from .config import load_config

        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
