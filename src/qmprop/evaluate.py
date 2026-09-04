"""Metrics and plots (Ch 3.13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def parity_plot(
    y_true,
    y_pred,
    title: str,
    path: str | Path,
    unit: str = "log S (mol/L)",
) -> Path:
    """Predicted vs. measured, with a y=x line and metrics in the corner.

    The standard figure for a regression paper: points hugging the
    diagonal is the whole claim, and systematic curvature away from it
    is visible here and invisible in a single RMSE number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = regression_metrics(y_true, y_pred)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.plot(lims, lims, color="#9A3350", lw=1, ls="--", zorder=1)
    ax.scatter(
        y_true, y_pred, s=18, alpha=0.6,
        color="#2C6E8F", edgecolor="none", zorder=2,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(f"measured {unit}")
    ax.set_ylabel(f"predicted {unit}")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.04, 0.96,
        f"RMSE {m['rmse']:.3f}\nMAE  {m['mae']:.3f}\nR²   {m['r2']:.3f}\nn    {m['n']}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
    )
    ax.grid(alpha=0.15, lw=0.5)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def format_table(rows: list[dict], columns: list[str]) -> str:
    """Fixed-width text table -- the ablation result, printable to a log."""
    widths = {
        c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in columns
    }
    line = "  ".join(c.ljust(widths[c]) for c in columns)
    rule = "  ".join("-" * widths[c] for c in columns)
    body = [
        "  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in columns)
        for r in rows
    ]
    return "\n".join([line, rule, *body])
