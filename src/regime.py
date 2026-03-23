"""
Market regime detection using Hidden Markov Models.

Identifies risk-on, risk-off, and crisis regimes from credit spread
dynamics using a Gaussian HMM fitted on rolling analytics.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_hmm_features(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.DatetimeIndex, StandardScaler]:
    """
    Select and standardize features for HMM input.

    Args:
        df: Enriched master DataFrame.
        feature_names: Column names to use as HMM inputs.

    Returns:
        Tuple of (X_scaled array, aligned DatetimeIndex, fitted scaler).

    Raises:
        KeyError: If any feature_name is not in df.
    """
    # Validate features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features in DataFrame: {missing}")

    # Select and drop NaN
    X_raw = df[feature_names].copy()
    rows_before = len(X_raw)
    X_raw = X_raw.dropna()
    rows_dropped = rows_before - len(X_raw)
    if rows_dropped > 0:
        logger.info("Dropped %d NaN rows from HMM features", rows_dropped)

    index = X_raw.index

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)

    logger.info(
        "Prepared HMM features: %d observations x %d features",
        X_scaled.shape[0], X_scaled.shape[1],
    )

    return X_scaled, index, scaler



def fit_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int = 200,
    n_init: int = 50,
    covariance_type: str = "full",
    random_state: int = 42,
) -> GaussianHMM:
    """
    Fit a Gaussian HMM on the prepared features.

    Args:
        X: Scaled feature array (n_samples, n_features).
        n_states: Number of hidden states.
        n_iter: Max EM iterations.
        n_init: Number of random initializations (best is kept).
        covariance_type: Covariance matrix type ("full", "diag", "tied").
        random_state: Random seed for reproducibility.

    Returns:
        Fitted GaussianHMM model.
    """
    logger.info(
        "Fitting GaussianHMM: %d states, %d features, %d observations...",
        n_states, X.shape[1], X.shape[0],
    )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    # Check convergence
    if hasattr(model, "monitor_") and hasattr(model.monitor_, "converged"):
        if model.monitor_.converged:
            logger.info("  HMM converged after %d iterations", model.monitor_.iter)
        else:
            logger.warning("  HMM did NOT converge after %d iterations", n_iter)
    
    log_likelihood = model.score(X)
    logger.info("  Log-likelihood: %.2f", log_likelihood)

    return model






