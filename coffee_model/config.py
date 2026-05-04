"""Shared configuration: paths, horizons, commodities, model hyperparameters."""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)

DATA_DIR = REPO_DIR
OUTPUT_DIR = REPO_DIR

HORIZONS = [30, 90, 180]
COMMODITIES = ["arabica_price", "robusta_price"]
QUANTILES = [0.05, 0.50, 0.95]

FIG_SIZE = (15, 8)

# Single source of truth for model hyperparameters. The walk-forward backtest,
# the held-out validation fit and the final live model all read from this dict
# so the three stay in sync.
MODEL_CONFIGS = {
    "arabica_price": {
        "sklearn": dict(n_estimators=200, learning_rate=0.05, max_depth=5,
                        random_state=42),
        "lightgbm": dict(n_estimators=400, learning_rate=0.04, num_leaves=31,
                         min_data_in_leaf=20, feature_fraction=0.9,
                         bagging_fraction=0.9, bagging_freq=5, verbosity=-1,
                         random_state=42),
    },
    "robusta_price": {
        "sklearn": dict(n_estimators=250, learning_rate=0.03, max_depth=4,
                        min_samples_leaf=20, random_state=42),
        "lightgbm": dict(n_estimators=500, learning_rate=0.025, num_leaves=20,
                         min_data_in_leaf=30, feature_fraction=0.85,
                         bagging_fraction=0.85, bagging_freq=5, verbosity=-1,
                         random_state=42),
    },
}

# Default engine: "sklearn" (GradientBoostingRegressor) keeps v5 baseline,
# "lightgbm" enables the new engine. Override with COFFEE_MODEL_ENGINE env var.
DEFAULT_ENGINE = os.getenv("COFFEE_MODEL_ENGINE", "lightgbm")
