"""Model registry (Ch 3 and 4).

Every model is wrapped in a Pipeline with a scaler, because half of these
(ridge, SVM, MLP) are scale-sensitive and half (the trees) are not --
and forgetting the scaler on the first group is the single most common
way to make a linear baseline look worse than it is.

The MLP here is sklearn's, not Keras or PyTorch. That is deliberate for
a scaffold: it runs everywhere with no extra dependency, and it is an
honest stand-in for the Ch 4.5 network. Swap in torch when you want the
GNN -- `build_model` is the only place that needs to change.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def build_model(name: str, seed: int = 42):
    """Return an unfitted estimator by name."""
    if name == "ridge":
        return _scaled(Ridge(alpha=1.0))

    if name == "lasso":
        return _scaled(Lasso(alpha=0.01, max_iter=10_000))

    if name == "elastic_net":
        return _scaled(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000))

    if name == "svr":
        return _scaled(SVR(C=10.0, epsilon=0.1, kernel="rbf"))

    if name == "random_forest":
        # No scaler: trees split on thresholds, so scale is irrelevant.
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=1,
            n_jobs=-1, random_state=seed,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed; `pip install xgboost` or drop "
                "'xgboost' from config.yaml models"
            ) from exc
        return XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=seed,
            tree_method="hist",
        )

    if name == "mlp":
        return _scaled(
            MLPRegressor(
                hidden_layer_sizes=(512, 128),
                activation="relu",
                alpha=1e-4,                # L2, the Ch 3.3 idea reused
                learning_rate_init=1e-3,
                max_iter=600,
                early_stopping=True,       # Ch 4.3
                n_iter_no_change=25,
                validation_fraction=0.1,
                random_state=seed,
            )
        )

    raise ValueError(f"unknown model {name!r}")


AVAILABLE = [
    "ridge", "lasso", "elastic_net", "svr",
    "random_forest", "xgboost", "mlp",
]
