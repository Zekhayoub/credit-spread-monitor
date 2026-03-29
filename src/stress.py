"""
Historical stress testing engine.

Identifies stress episodes from market conditions and measures
the forward impact on credit spreads across multiple horizons.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


def identify_trigger_dates(
    condition: pd.Series,
    cooldown: int = 20,
) -> list[pd.Timestamp]:
    """
    Identify dates where a stress condition is True, with cooldown.

    The cooldown prevents counting the same event multiple times.
    After a trigger, subsequent triggers within `cooldown` days
    are ignored.

    Args:
        condition: Boolean Series indexed by date.
        cooldown: Minimum days between triggers.

    Returns:
        List of trigger timestamps.
    """
    trigger_dates = []
    last_trigger = None

    for date, is_trigger in condition.items():
        if not is_trigger:
            continue
        if last_trigger is not None and (date - last_trigger).days < cooldown:
            continue
        trigger_dates.append(date)
        last_trigger = date

    logger.info("Found %d trigger dates (cooldown=%dd)", len(trigger_dates), cooldown)
    return trigger_dates





def compute_forward_impact(
    df: pd.DataFrame,
    trigger_dates: list[pd.Timestamp],
    spread_col: str,
    forward_windows: list[int],
) -> pd.DataFrame:
    """
    Measure spread impact after each trigger date.

    Computes both point-to-point AND path-based metrics:
    - final_impact: spread[T+N] - spread[T] (can mask intra-window extremes)
    - max_widening: worst spread widening within the window
    - max_compression: best spread tightening within the window
    - time_to_peak: trading days from trigger to worst widening
    - relative_impact: final_impact / spread_at_trigger (for cross-rating comparison)

    The max_widening is what actually triggers margin calls and forced selling.
    A fund doesn't survive to T+60 if it blows up at T+15.

    Args:
        df: DataFrame with spread data.
        trigger_dates: List of stress event dates.
        spread_col: Spread column to measure.
        forward_windows: List of forward horizons in trading days.

    Returns:
        DataFrame with one row per (episode, window) combination.
    """
    records = []

    for trigger in trigger_dates:
        if trigger not in df.index:
            continue

        trigger_loc = df.index.get_loc(trigger)
        spread_at_trigger = df[spread_col].iloc[trigger_loc]

        for window in forward_windows:
            end_loc = min(trigger_loc + window, len(df) - 1)
            if trigger_loc + window >= len(df):
                continue  # skip if not enough data

            # Extract the path within the window
            path = df[spread_col].iloc[trigger_loc:end_loc + 1]
            path_changes = path - spread_at_trigger

            final_impact = path_changes.iloc[-1]
            max_widening = path_changes.max()
            max_compression = path_changes.min()
            
            # Time to peak widening (in trading days)
            time_to_peak = path_changes.idxmax()
            time_to_peak_days = (time_to_peak - trigger).days if pd.notna(time_to_peak) else None

            # Relative impact (for cross-rating comparison)
            relative_impact = final_impact / spread_at_trigger if spread_at_trigger != 0 else np.nan
            relative_max_widening = max_widening / spread_at_trigger if spread_at_trigger != 0 else np.nan

            records.append({
                "trigger_date": trigger,
                "spread_col": spread_col,
                "window": window,
                "spread_at_trigger": spread_at_trigger,
                "final_impact": final_impact,
                "max_widening": max_widening,
                "max_compression": max_compression,
                "time_to_peak_days": time_to_peak_days,
                "relative_final_impact": relative_impact,
                "relative_max_widening": relative_max_widening,
            })

    return pd.DataFrame(records)



def run_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    condition: pd.Series,
    spread_cols: list[str],
    config: dict = CONFIG,
) -> pd.DataFrame:
    """
    Run a single stress test scenario.

    Args:
        df: Enriched DataFrame.
        scenario_name: Human-readable name (e.g., "VIX > 30").
        condition: Boolean Series of trigger dates.
        spread_cols: Spreads to measure impact on.
        config: Configuration dict.

    Returns:
        DataFrame with impact results for all spreads and windows.
    """
    stress_cfg = config["stress"]
    cooldown = stress_cfg["cooldown_days"]
    forward_windows = stress_cfg["forward_windows"]
    min_episodes = stress_cfg["min_episodes"]

    trigger_dates = identify_trigger_dates(condition, cooldown)

    if len(trigger_dates) == 0:
        logger.warning("Scenario '%s': 0 trigger dates. Skipping.", scenario_name)
        return pd.DataFrame()

    if len(trigger_dates) < min_episodes:
        logger.warning(
            "Scenario '%s': only %d episodes (min=%d). Results may not be statistically significant.",
            scenario_name, len(trigger_dates), min_episodes,
        )

    all_impacts = []
    for spread_col in spread_cols:
        impacts = compute_forward_impact(df, trigger_dates, spread_col, forward_windows)
        impacts["scenario"] = scenario_name
        all_impacts.append(impacts)

    result = pd.concat(all_impacts, ignore_index=True)
    logger.info("Scenario '%s': %d episodes, %d impact measurements",
                scenario_name, len(trigger_dates), len(result))

    return result



def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Array of observations.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    if len(values) < 3:
        return np.mean(values), np.nan, np.nan

    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(means, alpha * 100)
    ci_upper = np.percentile(means, (1 - alpha) * 100)

    return np.mean(values), ci_lower, ci_upper




def run_all_stress_tests(config: dict = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all stress test scenarios and compile results.

    Scenarios:
    - VIX level triggers: VIX > [25, 30, 35]
    - VIX shock triggers: delta VIX > [8, 10, 15] over 5 days
      (These capture the SHOCK, not the level — more actionable)
    - HY widening: hy_spread 20d change > threshold
    - Rate shock: treasury_10y 20d change > threshold

    Returns:
        Tuple of (detailed episodes DataFrame, summary stats DataFrame).
    """
    proc_dir = PROJECT_ROOT / config["paths"]["processed"]
    results_dir = PROJECT_ROOT / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proc_dir / "master_with_regimes.csv", index_col=0, parse_dates=True)
    spread_cols = ["bbb_spread", "hy_spread"]
    stress_cfg = config["stress"]

    all_episodes = []

    # --- VIX level triggers ---
    for threshold in stress_cfg["vix_thresholds"]:
        condition = df["vix"] > threshold
        result = run_scenario(df, f"VIX_level>{threshold}", condition, spread_cols, config)
        all_episodes.append(result)

    # --- VIX shock triggers (delta) ---
    # The VIX level is a lagging indicator — by the time it hits 30,
    # spreads have already widened. The SHOCK (rate of change) is what matters.
    vix_shock_thresholds = [8, 10, 15]
    df["vix_change_5d"] = df["vix"].diff(5)
    for threshold in vix_shock_thresholds:
        condition = df["vix_change_5d"] > threshold
        result = run_scenario(df, f"VIX_shock_5d>{threshold}", condition, spread_cols, config)
        all_episodes.append(result)

    # --- HY widening ---
    if "hy_spread_change_20d" in df.columns:
        condition = df["hy_spread_change_20d"] > stress_cfg["hy_widening_20d"]
        result = run_scenario(df, "HY_widening_20d", condition, spread_cols, config)
        all_episodes.append(result)

    # --- Rate shock ---
    if "treasury_10y" in df.columns:
        df["treasury_10y_change_20d"] = df["treasury_10y"].diff(20)
        condition = df["treasury_10y_change_20d"] > stress_cfg["yield_rise_20d"]
        result = run_scenario(df, "Rate_shock_20d", condition, spread_cols, config)
        all_episodes.append(result)

    # Combine
    episodes = pd.concat(all_episodes, ignore_index=True)

    # Summary statistics
    summary = (
        episodes
        .groupby(["scenario", "spread_col", "window"])
        .agg(
            n_episodes=("final_impact", "count"),
            mean_final_impact=("final_impact", "mean"),
            median_final_impact=("final_impact", "median"),
            mean_max_widening=("max_widening", "mean"),
            median_max_widening=("max_widening", "median"),
            mean_relative_impact=("relative_final_impact", "mean"),
            mean_relative_max_widening=("relative_max_widening", "mean"),
        )
        .reset_index()
    )

    # Add bootstrap CI to summary
    ci_records = []
    for _, group in episodes.groupby(["scenario", "spread_col", "window"]):
        values = group["max_widening"].dropna().values
        
        if len(values) == 0:
            continue
            
        mean_val, ci_low, ci_high = bootstrap_confidence_interval(values)
        ci_records.append({
            "scenario": group["scenario"].iloc[0],
            "spread_col": group["spread_col"].iloc[0],
            "window": group["window"].iloc[0],
            "n_episodes": len(values),
            "max_widening_mean": mean_val,
            "max_widening_ci_lower": ci_low,
            "max_widening_ci_upper": ci_high,
            "significant": len(values) >= 10,  # flag low-sample results
        })

    ci_df = pd.DataFrame(ci_records)

    # Save
    episodes.to_csv(results_dir / "stress_episodes.csv", index=False)
    summary.to_csv(results_dir / "stress_test_results.csv", index=False)
    ci_df.to_csv(results_dir / "stress_test_confidence.csv", index=False)

    logger.info("Stress testing complete: %d total episodes across all scenarios", len(episodes))

    return episodes, summary



def cooldown_sensitivity(
    df: pd.DataFrame,
    condition: pd.Series,
    cooldowns: list[int] = [10, 20, 40],
) -> pd.DataFrame:
    """
    Test sensitivity of trigger count to cooldown parameter.

    Args:
        df: DataFrame.
        condition: Boolean trigger condition.
        cooldowns: List of cooldown values to test.

    Returns:
        DataFrame with cooldown value and number of episodes.
    """
    records = []
    for cd in cooldowns:
        triggers = identify_trigger_dates(condition, cooldown=cd)
        records.append({"cooldown_days": cd, "n_episodes": len(triggers)})

    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    episodes, summary = run_all_stress_tests()
    print(f"\nStress testing complete")
    print(f"Total episodes: {len(episodes)}")
    print(f"\nSummary:\n{summary.to_string()}")

    

