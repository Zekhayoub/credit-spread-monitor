"""
Feature engineering for credit spread analysis.

Computes rolling analytics used by fixed income professionals:
z-scores, percentiles, realized volatility, compression ratios.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Spread columns (used across multiple functions)
SPREAD_COLS = ["aaa_spread", "aa_spread", "bbb_spread", "hy_spread"]


def compute_spread_changes(
    df: pd.DataFrame,
    spread_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """
    Compute absolute spread changes over multiple horizons.

    Args:
        df: Master DataFrame.
        spread_col: Column name (e.g., "bbb_spread").
        windows: List of lookback periods in trading days.

    Returns:
        DataFrame with new columns: {spread_col}_change_{N}d
    """
    if spread_col not in df.columns:
        raise KeyError(f"Column {spread_col} not found in DataFrame")

    result = pd.DataFrame(index=df.index)
    for w in windows:
        col_name = f"{spread_col}_change_{w}d"
        result[col_name] = df[spread_col].diff(w)

    return result


def compute_zscore(
    series: pd.Series,
    window: int,
    min_std: float = 1e-8,
) -> pd.Series:
    """
    Rolling z-score: (value - rolling_mean) / rolling_std.

    Guards against division by zero and near-zero std that produce
    extreme z-scores when spreads are flat.

    Args:
        series: Input time series.
        window: Rolling window in trading days.
        min_std: Minimum std threshold. Below this, z-score is set to 0.

    Returns:
        pd.Series of z-scores.
    """
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window, min_periods=window // 2).std()

    zscore = np.where(
        rolling_std > min_std,
        (series - rolling_mean) / rolling_std,
        0.0,
    )

    result = pd.Series(zscore, index=series.index)

    # Assertion: no inf values
    assert not np.isinf(result).any(), "Inf values in z-score — check min_std threshold"

    return result



def compute_percentile(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Rolling percentile rank (0 to 1).

    Unlike z-scores, percentiles make no distributional assumptions.
    They are the primary metric for assessing whether spreads are
    historically wide or tight.

    A percentile of 0.90 means the current spread is wider than 90%
    of observations in the lookback window.

    Args:
        series: Input time series.
        window: Rolling window in trading days.

    Returns:
        pd.Series between 0.0 and 1.0.
    """
    # Vectorized — uses compiled C under the hood
    # ~100x faster than .rolling().apply(lambda x: ...)
    pctile = series.rolling(window, min_periods=window // 2).rank() / window

    return pctile



def compute_rolling_volatility(
    spread_changes: pd.Series,
    window: int,
    trading_mask: pd.Series | None = None,
) -> pd.Series:
    """
    Rolling realized volatility, computed on real trading days only.

    If a trading_mask is provided, forward-filled days (weekends,
    holidays) are excluded from the volatility calculation to avoid
    artificial zero-change days that compress the rolling std.

    Args:
        spread_changes: Daily spread changes (diff(1)).
        window: Rolling window in trading days.
        trading_mask: Boolean Series — True for real trading days.
                      If None, all days are used (less accurate).

    Returns:
        pd.Series of rolling volatility.
    """
    changes = spread_changes.copy()

    if trading_mask is not None:
        # Set non-trading day changes to NaN so they don't contribute
        # zero-change observations that artificially compress volatility
        changes = changes.where(trading_mask)
        logger.debug("Volatility: excluded %d non-trading days", (~trading_mask).sum())

    vol = changes.rolling(window, min_periods=window // 2).std()
    return vol


def compute_compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread compression ratios.

    bbb_hy_ratio: bbb_spread / hy_spread
        Measures the crossover boundary between low IG and HY.
        Note: this is NOT a full IG/HY ratio (IG spans AAA to BBB).
        When this ratio falls -> HY widens faster -> risk-off.
        When it rises -> spreads converge -> compression / risk-on.

    aa_bbb_ratio: aa_spread / bbb_spread
        Measures compression within Investment Grade.

    Args:
        df: DataFrame with spread columns.

    Returns:
        DataFrame with ratio columns.
    """
    result = pd.DataFrame(index=df.index)

    if "bbb_spread" in df.columns and "hy_spread" in df.columns:
        result["bbb_hy_ratio"] = np.where(
            df["hy_spread"] != 0,
            df["bbb_spread"] / df["hy_spread"],
            np.nan,
        )
        result["bbb_hy_ratio_change_1d"] = pd.Series(
            result["bbb_hy_ratio"]
        ).diff(1)

    if "aa_spread" in df.columns and "bbb_spread" in df.columns:
        result["aa_bbb_ratio"] = np.where(
            df["bbb_spread"] != 0,
            df["aa_spread"] / df["bbb_spread"],
            np.nan,
        )

    return result

def apply_rolling_analytics(
    df: pd.DataFrame,
    spread_col: str,
    trading_mask: pd.DataFrame | None,
    config: dict = CONFIG,
) -> pd.DataFrame:
    """
    Apply all rolling analytics to a single spread column.

    Computes changes, z-scores, percentiles, and volatility for one spread.

    Args:
        df: Master DataFrame.
        spread_col: Spread column name (e.g., "bbb_spread").
        trading_mask: Boolean DataFrame of real trading days.
        config: Configuration dict.

    Returns:
        DataFrame with all computed features for this spread.
    """
    features = config["features"]
    result = pd.DataFrame(index=df.index)

    # Spread changes
    for w in features["change_windows"]:
        result[f"{spread_col}_change_{w}d"] = df[spread_col].diff(w)

    # Z-scores
    for w in features["zscore_windows"]:
        result[f"{spread_col}_zscore_{w}d"] = compute_zscore(df[spread_col], w)

    # Percentiles
    for w in features["percentile_windows"]:
        result[f"{spread_col}_pctile_{w}d"] = compute_percentile(df[spread_col], w)

    # Volatility (on trading days only)
    changes_1d = df[spread_col].diff(1)
    mask_col = trading_mask[spread_col] if trading_mask is not None else None
    for w in features["volatility_windows"]:
        result[f"{spread_col}_rolling_vol_{w}d"] = compute_rolling_volatility(
            changes_1d, w, trading_mask=mask_col,
        )

    return result


def enrich_master(
    df: pd.DataFrame,
    trading_mask: pd.DataFrame | None = None,
    config: dict = CONFIG,
) -> pd.DataFrame:
    """
    Apply all feature engineering to the master DataFrame.

    Steps:
        1. Rolling analytics for each spread (changes, z-scores, percentiles, vol)
        2. Compression ratios (bbb_hy, aa_bbb)
        3. Drop warm-up NaN rows
        4. Quality assertions

    Args:
        df: Master DataFrame from ingestion (8 cols, ~6000 rows).
        trading_mask: Boolean DataFrame of real trading days.
        config: Configuration dict.

    Returns:
        Enriched DataFrame with ~50 columns.
    """
    logger.info("Starting feature engineering...")
    result = df.copy()

    # 1. Rolling analytics per spread
    spread_cols = [c for c in SPREAD_COLS if c in df.columns]
    for col in spread_cols:
        analytics = apply_rolling_analytics(df, col, trading_mask, config)
        result = pd.concat([result, analytics], axis=1)
        logger.info("  Computed analytics for %s (%d features)", col, len(analytics.columns))

    # 2. Compression ratios
    ratios = compute_compression_ratio(df)
    result = pd.concat([result, ratios], axis=1)
    logger.info("  Computed compression ratios (%d features)", len(ratios.columns))

    # 3. Drop warm-up NaN rows (rolling windows need history)
    max_window = max(
        max(config["features"]["zscore_windows"]),
        max(config["features"]["percentile_windows"]),
        max(config["features"]["volatility_windows"]),
    )
    rows_before = len(result)
    result = result.dropna()
    rows_dropped = rows_before - len(result)
    logger.info("  Dropped %d warm-up rows (max window: %dd)", rows_dropped, max_window)

    # 4. Quality assertions
    assert result.isna().sum().sum() == 0, "NaN remaining after feature engineering"
    assert not np.isinf(result.values).any(), "Inf values in enriched DataFrame"
    assert len(result) > 1000, f"Only {len(result)} rows after enrichment"

    n_new = len(result.columns) - len(df.columns)
    logger.info(
        "Feature engineering complete: %d rows x %d columns (%d new features)",
        len(result), len(result.columns), n_new,
    )

    return result


def run_features(config: dict = CONFIG) -> pd.DataFrame:
    """
    Main entry point for feature engineering.

    Loads master.csv + trading_mask.csv, enriches, saves master_enriched.csv.

    Returns:
        Enriched DataFrame.
    """
    proc_dir = PROJECT_ROOT / config["paths"]["processed"]

    # Load master
    master_path = proc_dir / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Master not found: {master_path}. Run ingestion first.")
    df = pd.read_csv(master_path, index_col=0, parse_dates=True)

    # Load trading mask
    mask_path = proc_dir / "trading_mask.csv"
    trading_mask = None
    if mask_path.exists():
        trading_mask = pd.read_csv(mask_path, index_col=0, parse_dates=True).astype(bool)
        # Align to df index
        trading_mask = trading_mask.loc[df.index]
        logger.info("Loaded trading mask")
    else:
        logger.warning("Trading mask not found — volatility will include non-trading days")

    # Enrich
    enriched = enrich_master(df, trading_mask, config)

    # Save
    output_path = proc_dir / "master_enriched.csv"
    enriched.to_csv(output_path)
    logger.info("Saved enriched DataFrame: %s", output_path)

    return enriched


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    df = run_features()
    print(f"\nEnriched DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns:\n{list(df.columns)}")





