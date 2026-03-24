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
    min_history: int = 252,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Select and standardize features for HMM input using expanding window.

    IMPORTANT: Uses expanding (cumulative) mean and std instead of a global
    StandardScaler to avoid look-ahead bias. At each date t, normalization
    uses only data from [0, t], never future observations.

    Args:
        df: Enriched master DataFrame.
        feature_names: Column names to use as HMM inputs.
        min_history: Minimum observations before scaling starts.
                     Earlier rows are dropped (not enough history to normalize).

    Returns:
        Tuple of (X_scaled array, aligned DatetimeIndex).
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

    # Expanding window standardization (no look-ahead bias)
    expanding_mean = X_raw.expanding(min_periods=min_history).mean()
    expanding_std = X_raw.expanding(min_periods=min_history).std()

    # Replace zero std with small value to avoid division by zero
    expanding_std = expanding_std.replace(0, 1e-8)

    X_scaled_df = (X_raw - expanding_mean) / expanding_std

    # Drop rows where expanding window hasn't accumulated enough history
    X_scaled_df = X_scaled_df.dropna()

    index = X_scaled_df.index
    X_scaled = X_scaled_df.values

    logger.info(
        "Prepared HMM features (expanding window, no look-ahead): "
        "%d observations x %d features (dropped %d for warm-up)",
        X_scaled.shape[0], X_scaled.shape[1],
        rows_before - len(X_scaled_df),
    )

    return X_scaled, index



def fit_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int = 200,
    n_init: int = 50,
    covariance_type: str = "full",
    random_state: int = 42,
) -> GaussianHMM:
    """
    Fit a Gaussian HMM with multiple random initializations.

    Runs n_init fits with different random seeds and keeps the model
    with the highest log-likelihood. This stabilizes results that would
    otherwise vary between runs due to EM's sensitivity to initialization.

    Args:
        X: Scaled feature array.
        n_states: Number of hidden states.
        n_iter: Max EM iterations per fit.
        n_init: Number of random initializations.
        covariance_type: Covariance type.
        random_state: Base random seed.

    Returns:
        Best fitted GaussianHMM (highest log-likelihood).
    """
    logger.info(
        "Fitting GaussianHMM: %d states, %d inits, %d features, %d obs...",
        n_states, n_init, X.shape[1], X.shape[0],
    )

    best_model = None
    best_score = -np.inf

    for i in range(n_init):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state + i,
            )
            model.fit(X)
            score = model.score(X)

            if score > best_score:
                best_score = score
                best_model = model

        except Exception:
            continue  # some inits may fail, that's expected

    if best_model is None:
        raise RuntimeError(f"All {n_init} initializations failed for {n_states} states")

    converged = (hasattr(best_model, "monitor_") and 
                 hasattr(best_model.monitor_, "converged") and 
                 best_model.monitor_.converged)
    
    logger.info(
        "  Best of %d inits: log_lik=%.2f, converged=%s",
        n_init, best_score, converged,
    )

    return best_model




def compute_n_params(n_states: int, n_features: int, covariance_type: str = "full") -> int:
    """
    Compute the exact number of free parameters in a Gaussian HMM.

    For 3 states and 4 features with full covariance:
    - Transition matrix: 3 * (3-1) = 6 free params
    - Means: 3 * 4 = 12 params
    - Covariances (symmetric): 3 * 4 * (4+1) / 2 = 30 params
    - Initial state probs: 3 - 1 = 2 params
    - Total: 50 params

    Args:
        n_states: Number of hidden states.
        n_features: Number of observation features.
        covariance_type: "full", "diag", or "tied".

    Returns:
        Number of free parameters.
    """
    # Transition matrix: each row sums to 1, so n_states - 1 free per row
    n_transition = n_states * (n_states - 1)

    # Initial state distribution: n_states - 1 free params
    n_initial = n_states - 1

    # Means: n_states * n_features
    n_means = n_states * n_features

    # Covariance matrices (symmetric positive definite)
    if covariance_type == "full":
        # Each state has a full symmetric matrix: n_features * (n_features + 1) / 2
        n_cov = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        n_cov = n_states * n_features
    elif covariance_type == "tied":
        n_cov = n_features * (n_features + 1) // 2
    else:
        n_cov = n_states * n_features  # fallback to diag

    total = n_transition + n_initial + n_means + n_cov
    
    logger.debug(
        "HMM params (n=%d, f=%d, %s): transition=%d, initial=%d, means=%d, cov=%d, total=%d",
        n_states, n_features, covariance_type,
        n_transition, n_initial, n_means, n_cov, total,
    )

    return total


def select_n_states_bic(
    X: np.ndarray,
    n_range: list[int],
    config: dict = CONFIG,
) -> tuple[int, dict]:
    """
    Select optimal number of HMM states using BIC.

    BIC = -2 * log_likelihood + n_params * log(n_observations)

    Args:
        X: Scaled feature array.
        n_range: List of state counts to test.
        config: Configuration dict.

    Returns:
        Tuple of (optimal n_states, dict of {n: bic_value}).
    """
    regime_cfg = config["regime"]
    n_obs = X.shape[0]
    n_features = X.shape[1]
    results = {}

    for n in n_range:
        try:
            model = fit_hmm(
                X, n_states=n,
                n_iter=regime_cfg["n_iter"],
                n_init=regime_cfg.get("n_init", 10),
                covariance_type=regime_cfg["covariance_type"],
                random_state=regime_cfg["random_state"],
            )
            
            log_likelihood = model.score(X) * n_obs  # score() returns per-sample
            n_params = compute_n_params(n, n_features, regime_cfg["covariance_type"])
            bic = -2 * log_likelihood + n_params * np.log(n_obs)
            
            results[n] = bic
            logger.info("  n_states=%d: BIC=%.2f (log_lik=%.2f, n_params=%d)",
                        n, bic, log_likelihood, n_params)

        except Exception as e:
            logger.warning("  n_states=%d: failed (%s)", n, e)
            results[n] = np.inf

    # Select best
    best_n = min(results, key=results.get)
    
    # Fallback if BIC is inconclusive (all very close)
    bic_values = [v for v in results.values() if v != np.inf]
    if len(bic_values) > 1:
        bic_range = max(bic_values) - min(bic_values)
        if bic_range < 10:  # arbitrary threshold for "too close"
            best_n = config["regime"]["n_states_default"]
            logger.warning(
                "BIC values too close (range=%.2f). Falling back to default: %d states",
                bic_range, best_n,
            )

    logger.info("Selected %d states (BIC=%.2f)", best_n, results[best_n])
    return best_n, results



def label_regimes(
    model: GaussianHMM,
    X: np.ndarray,
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Predict regimes and assign interpretable labels.

    First version: labels based on mean bbb_spread per state.
    Lowest mean = risk_on, highest = crisis.

    Args:
        model: Fitted GaussianHMM.
        X: Scaled feature array.
        df: Original enriched DataFrame.
        index: DatetimeIndex aligned to X.

    Returns:
        DataFrame with 'regime' and 'regime_proba' columns added.
    """
    # Predict states and probabilities
    states = model.predict(X)
    state_probs = model.predict_proba(X)
    max_probs = state_probs.max(axis=1)

    # Compute mean bbb_spread per state for labeling
    df_aligned = df.loc[index].copy()
    df_aligned["state"] = states

    state_means = df_aligned.groupby("state")["bbb_spread"].mean().sort_values()
    
    # Map: lowest mean -> risk_on, middle -> risk_off, highest -> crisis
    n_states = model.n_components
    if n_states == 3:
        label_map = {
            state_means.index[0]: "risk_on",
            state_means.index[1]: "risk_off",
            state_means.index[2]: "crisis",
        }
    elif n_states == 2:
        label_map = {
            state_means.index[0]: "risk_on",
            state_means.index[1]: "crisis",
        }
    else:
        # Generic labeling for 4+ states
        label_map = {}
        for i, state_idx in enumerate(state_means.index):
            if i == 0:
                label_map[state_idx] = "risk_on"
            elif i == len(state_means) - 1:
                label_map[state_idx] = "crisis"
            else:
                label_map[state_idx] = f"risk_off_{i}"

    # Apply labels
    df_aligned["regime"] = df_aligned["state"].map(label_map)
    df_aligned["regime_proba"] = max_probs

    logger.info("Regime distribution:")
    for regime, count in df_aligned["regime"].value_counts().items():
        pct = count / len(df_aligned) * 100
        logger.info("  %s: %d days (%.1f%%)", regime, count, pct)

    return df_aligned



