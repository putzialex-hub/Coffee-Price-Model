"""Hyperparameter tuning for tail quantile models via TimeSeriesSplit + pinball CV.

Usage (from main.py CLI):  python -m coffee_model.main --tune
Writes optimal TAIL_OVERRIDES to model_configs_tuned.json (TTL: 14 days).
On the next run the cache is loaded automatically — no re-tuning needed.

Why tune the tails separately?
  The default LightGBM params (min_data_in_leaf=20, num_leaves=31) allow the
  model to fit too locally, causing q=0.05 and q=0.95 to converge toward q=0.50.
  Result: raw coverage ~20-47% for a nominal 90% interval.  Increasing
  min_data_in_leaf and reducing num_leaves forces smoother tail predictions
  (wider intervals) without hurting point-forecast accuracy.

Grid searched: min_data_in_leaf × num_leaves for tail quantiles.
Objective: minimise mean pinball loss across 5 time-series folds.
"""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime

import numpy as np

from .config import (
    COMMODITIES, HORIZONS, MODEL_CONFIGS, QUANTILES, REPO_DIR, TAIL_OVERRIDES,
)
from .features import all_feature_columns, fill_remaining_nan_with_zero, get_commodity_features
from .validation import _pinball_loss

try:
    from sklearn.model_selection import TimeSeriesSplit
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

TUNE_CACHE_PATH = os.path.join(REPO_DIR, "model_configs_tuned.json")
TUNE_MAX_AGE_DAYS = 14

# Search space for tail-quantile hyperparameters.
#
# Key insight: to get wider raw intervals for q=0.05 / q=0.95, we need the
# tail model to make more *extreme* predictions, NOT smoother ones. Therefore:
#  • lower min_data_in_leaf → allows smaller leaves → more extreme local fits
#  • higher reg_lambda      → L2 regularisation controls overfitting separately
#  • lower learning_rate    → more trees to accumulate stable tail signal
# This is the opposite of what one might naively expect from regularisation theory.
TAIL_LGBM_GRID: dict[str, list] = {
    "min_data_in_leaf": [5, 10, 20],        # lower = more extreme tail fits
    "reg_lambda":       [0.0, 1.0, 5.0],   # L2 penalty for variance control
    "learning_rate":    [0.02, 0.04],       # slower = more accumulated signal
}
TAIL_SKLEARN_GRID: dict[str, list] = {
    "min_samples_leaf": [5, 10, 20],
    "max_depth":        [4, 5, 6],
}


def _cache_is_fresh() -> bool:
    if not os.path.exists(TUNE_CACHE_PATH):
        return False
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(TUNE_CACHE_PATH))).days
    return age < TUNE_MAX_AGE_DAYS


def load_tuned_overrides() -> bool:
    """Load cached TAIL_OVERRIDES into the live config. Returns True if loaded."""
    if not _cache_is_fresh():
        return False
    try:
        with open(TUNE_CACHE_PATH) as f:
            cached: dict = json.load(f)
        for comm, engines in cached.items():
            if comm not in TAIL_OVERRIDES:
                TAIL_OVERRIDES[comm] = {}
            for eng, params in engines.items():
                TAIL_OVERRIDES[comm][eng] = params
        print(f"   ✅ Tuned hyperparameters loaded from {os.path.basename(TUNE_CACHE_PATH)}")
        return True
    except Exception as e:
        print(f"   ⚠️ Tuning cache unreadable ({e}), using defaults.")
        return False


def _pinball_cv_loss(
    df_clean,
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    commodity: str,
    quantile: float,
    engine: str,
    override_params: dict,
    n_splits: int = 5,
) -> float:
    """Mean pinball loss across TimeSeriesSplit folds for given hyperparams."""
    from sklearn.model_selection import TimeSeriesSplit
    from .models import _build_estimator  # local import to avoid circular at module level

    # Temporarily update TAIL_OVERRIDES for this evaluation
    orig = TAIL_OVERRIDES.get(commodity, {}).get(engine, {}).copy()
    TAIL_OVERRIDES.setdefault(commodity, {})[engine] = override_params

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_losses: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        if len(train_idx) < 200 or len(val_idx) < 10:
            continue
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        p_va = prices[val_idx]

        est = _build_estimator(commodity, quantile, engine)
        est.fit(X_tr, y_tr)
        y_pred_log = est.predict(X_va)

        losses = [
            _pinball_loss(p_va[i] * np.exp(y_va[i]),
                          p_va[i] * np.exp(y_pred_log[i]),
                          quantile)
            for i in range(len(y_va))
        ]
        fold_losses.append(float(np.mean(losses)))

    # Restore original override
    if orig:
        TAIL_OVERRIDES[commodity][engine] = orig
    elif engine in TAIL_OVERRIDES.get(commodity, {}):
        del TAIL_OVERRIDES[commodity][engine]

    return float(np.mean(fold_losses)) if fold_losses else float("inf")


def tune_hyperparams(
    df,
    engine: str = "lightgbm",
    n_splits: int = 5,
    force: bool = False,
) -> None:
    """Grid-search optimal tail-quantile hyperparameters and update TAIL_OVERRIDES.

    Writes results to TUNE_CACHE_PATH so subsequent runs skip re-tuning.
    The live TAIL_OVERRIDES dict is updated in-place — no restart required.
    """
    if not _SKLEARN_AVAILABLE:
        print("   ⚠️ sklearn not found — cannot tune. Using default TAIL_OVERRIDES.")
        return
    if not force and _cache_is_fresh():
        load_tuned_overrides()
        return

    grid = TAIL_LGBM_GRID if engine == "lightgbm" else TAIL_SKLEARN_GRID
    tail_quantiles = [q for q in QUANTILES if q != 0.5]

    n_combos = len(list(itertools.product(*grid.values())))
    n_total = len(COMMODITIES) * len(HORIZONS) * len(tail_quantiles) * n_combos
    print(f"\n   🔍 Hyperparameter-Tuning ({n_total} model fits, ~{n_total // 10}s)...")

    df_work = fill_remaining_nan_with_zero(df) if engine == "sklearn" else df.copy()
    results: dict[str, dict[str, dict]] = {}
    fit_count = 0

    for comm in COMMODITIES:
        results[comm] = {}
        fcols = get_commodity_features(all_feature_columns(df_work), comm)

        for h in HORIZONS:
            target_col = f"_tune_{comm}_{h}d"
            df_work[target_col] = np.log(df_work[comm].shift(-h) / df_work[comm])
            clean = df_work.dropna(subset=[target_col, comm])
            X = clean[fcols].values
            y = clean[target_col].values
            prices = clean[comm].values

            # Determine best params shared across both tail quantiles
            # (we optimize average loss over q=0.05 and q=0.95 jointly for simplicity)
            best_loss = float("inf")
            best_params: dict = {}

            keys, vals_list = list(grid.keys()), list(grid.values())
            for combo in itertools.product(*vals_list):
                params = dict(zip(keys, combo))
                avg_loss = 0.0
                for q in tail_quantiles:
                    avg_loss += _pinball_cv_loss(
                        df_work, X, y, prices, comm, q, engine, params, n_splits
                    )
                    fit_count += n_splits
                avg_loss /= len(tail_quantiles)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_params = params

                print(f"      [{fit_count}/{n_total*n_splits}] "
                      f"{comm.replace('_price','')} h={h:3d}: "
                      f"{params} → pinball={avg_loss:.4f}", end="\r", flush=True)

            results[comm][str(h)] = best_params
            print(f"\n      ✓ {comm.replace('_price','')} {h}d best: {best_params} "
                  f"(pinball={best_loss:.4f})")

    # Build unified override (use the 180d params as the conservative default —
    # longer horizons need wider intervals most)
    tuned_overrides: dict[str, dict[str, dict]] = {}
    for comm in COMMODITIES:
        best_180 = results[comm].get(str(max(HORIZONS)), {})
        tuned_overrides[comm] = {engine: best_180}
        TAIL_OVERRIDES.setdefault(comm, {})[engine] = best_180

    with open(TUNE_CACHE_PATH, "w") as f:
        json.dump(tuned_overrides, f, indent=2)
    print(f"\n   ✅ Tuning abgeschlossen → {TUNE_CACHE_PATH}")
    print(f"   Neue TAIL_OVERRIDES: {tuned_overrides}")
