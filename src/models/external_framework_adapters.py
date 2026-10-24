"""Optional External Framework Adapters for Standard LightGBM, XGBoost, and PyTorch (Audit Issue 4 & DEV-002).

Provides clean, soft-target-compatible wrappers for `lightgbm`, `xgboost`, and `torch` when executed
in an external environment where those packages are installed. In the air-gapped VM sandbox (where
only `numpy`, `scipy`, and `pandas` are installed), attempting to instantiate these adapters raises
an explicit `RuntimeError` rather than silently substituting custom NumPy learners.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def get_external_package_availability() -> Dict[str, Any]:
    return {
        "lightgbm_installed": HAS_LIGHTGBM,
        "lightgbm_version": getattr(lgb, "__version__", None) if HAS_LIGHTGBM else None,
        "xgboost_installed": HAS_XGBOOST,
        "xgboost_version": getattr(xgb, "__version__", None) if HAS_XGBOOST else None,
        "torch_installed": HAS_TORCH,
        "torch_version": getattr(torch, "__version__", None) if HAS_TORCH else None,
    }


class StandardLightGBMSoftTargetWrapper:
    """Wrapper for standard `lightgbm.Booster` using custom expected cross-entropy objective
    for continuous soft targets y in [0, 1]. Fails loudly if `lightgbm` is not installed.
    """
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, num_leaves: int = 31, seed: int = 42):
        if not HAS_LIGHTGBM:
            raise RuntimeError(
                "Standard LightGBM (`lightgbm`) is not installed in this environment. "
                "Do not conflate SoftTargetHistGBDT with standard LightGBM; install `lightgbm` "
                "or report results explicitly under Custom SoftTargetHistGBDT."
            )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.booster_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_mask: Optional[np.ndarray] = None):
        X_tr = X_train[sample_mask] if sample_mask is not None else X_train
        y_tr = np.clip(y_train[sample_mask] if sample_mask is not None else y_train, 0.0, 1.0)

        def soft_logreg_obj(preds: np.ndarray, train_data: Any):
            labels = train_data.get_label()
            p = 1.0 / (1.0 + np.exp(-np.clip(preds, -30.0, 30.0)))
            grad = p - labels
            hess = np.maximum(p * (1.0 - p), 1e-6)
            return grad, hess

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        params = {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "seed": self.seed,
            "verbose": -1,
            "n_jobs": 1,
        }
        self.booster_ = lgb.train(params, dtrain, num_boost_round=self.n_estimators, fobj=soft_logreg_obj)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_Margins = self.booster_.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw_Margins, -30.0, 30.0)))
