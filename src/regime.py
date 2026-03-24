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

    Uses multi-dimensional criteria:
    - Mean spread level (high = stress)
    - Mean momentum (negative = compression/recovery, positive = widening)
    - Mean volatility (high = unstable)

    This avoids the naive labeling where a high-spread recovery period
    (like Q3 2020) gets mislabeled as "crisis" just because spread
    levels are still elevated.

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

    # Build state profiles
    df_aligned = df.loc[index].copy()
    df_aligned["state"] = states

    profiles = {}
    for state in sorted(df_aligned["state"].unique()):
        mask = df_aligned["state"] == state
        profiles[state] = {
            "mean_spread": df_aligned.loc[mask, "bbb_spread"].mean(),
            "mean_momentum": df_aligned.loc[mask, "bbb_spread_change_20d"].mean()
                             if "bbb_spread_change_20d" in df_aligned.columns else 0,
            "mean_vol": df_aligned.loc[mask, "bbb_spread_rolling_vol_20d"].mean()
                        if "bbb_spread_rolling_vol_20d" in df_aligned.columns else 0,
            "count": mask.sum(),
        }

    logger.info("State profiles:")
    for state, profile in profiles.items():
        logger.info(
            "  State %d: spread=%.3f, momentum=%.4f, vol=%.4f, n=%d",
            state, profile["mean_spread"], profile["mean_momentum"],
            profile["mean_vol"], profile["count"],
        )

    # Multi-dimensional labeling
    # Score each state: high spread + positive momentum + high vol = crisis
    # Low spread + low vol = risk_on
    # High spread + negative momentum = recovery (label as risk_off)
    state_scores = {}
    for state, p in profiles.items():
        # Composite score: higher = more stressed
        # Momentum sign matters: positive momentum (widening) adds to stress
        score = (
            p["mean_spread"] * 1.0
            + p["mean_momentum"] * 50.0   # amplify momentum signal
            + p["mean_vol"] * 10.0
        )
        state_scores[state] = score

    # Sort states by composite score
    sorted_states = sorted(state_scores, key=state_scores.get)

    n_states = len(sorted_states)
    if n_states == 3:
        label_map = {
            sorted_states[0]: "risk_on",
            sorted_states[1]: "risk_off",
            sorted_states[2]: "crisis",
        }
    elif n_states == 2:
        label_map = {
            sorted_states[0]: "risk_on",
            sorted_states[1]: "crisis",
        }
    else:
        label_map = {}
        for i, state in enumerate(sorted_states):
            if i == 0:
                label_map[state] = "risk_on"
            elif i == len(sorted_states) - 1:
                label_map[state] = "crisis"
            else:
                label_map[state] = f"transition_{i}"

    # Apply
    df_aligned["regime"] = df_aligned["state"].map(label_map)
    df_aligned["regime_proba"] = max_probs

    # Drop temporary state column
    df_aligned = df_aligned.drop(columns=["state"])

    logger.info("Regime distribution (final):")
    for regime, count in df_aligned["regime"].value_counts().items():
        pct = count / len(df_aligned) * 100
        logger.info("  %s: %d days (%.1f%%)", regime, count, pct)

    return df_aligned



def compute_transition_matrix(
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute empirical regime transition probabilities.

    Counts transitions regime[t] -> regime[t+1] and normalizes to probabilities.

    Args:
        regimes: Series of regime labels (e.g., "risk_on", "risk_off", "crisis").

    Returns:
        DataFrame: transition probability matrix (rows = from, columns = to).
    """
    labels = sorted(regimes.unique())
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)

    for (curr, nxt) in zip(regimes.iloc[:-1], regimes.iloc[1:]):
        matrix.loc[curr, nxt] += 1

    # Normalize rows to probabilities
    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums, axis=0)

    return matrix


def compute_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per regime.

    For each regime: mean/median/std of each spread, mean volatility,
    mean episode duration.

    Args:
        df: DataFrame with 'regime' column.

    Returns:
        DataFrame with one row per regime.
    """
    stats = []
    spread_cols = [c for c in df.columns if c.endswith("_spread")]
    vol_cols = [c for c in df.columns if "rolling_vol" in c]

    for regime in sorted(df["regime"].unique()):
        mask = df["regime"] == regime
        row = {"regime": regime, "n_days": mask.sum()}

        for col in spread_cols:
            row[f"{col}_mean"] = df.loc[mask, col].mean()
            row[f"{col}_median"] = df.loc[mask, col].median()
            row[f"{col}_std"] = df.loc[mask, col].std()

        for col in vol_cols[:2]:  # top 2 vol columns
            row[f"{col}_mean"] = df.loc[mask, col].mean()

        # Episode duration: count consecutive days in same regime
        regime_blocks = (df["regime"] != df["regime"].shift()).cumsum()
        regime_episodes = df.loc[mask].groupby(regime_blocks[mask]).size()
        row["mean_episode_days"] = regime_episodes.mean()
        row["median_episode_days"] = regime_episodes.median()
        row["n_episodes"] = len(regime_episodes)

        stats.append(row)

    return pd.DataFrame(stats)



