"""Config loading. One YAML file drives every script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and resolve output_dir to an absolute path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["project_root"] = PROJECT_ROOT
    cfg["output_dir"] = PROJECT_ROOT / cfg.get("output_dir", "results")
    cfg["data_dir"] = PROJECT_ROOT / "data"
    return cfg
