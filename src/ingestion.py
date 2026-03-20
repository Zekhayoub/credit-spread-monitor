"""
FRED API data ingestion.

Downloads credit spread series, Treasury yields, and market indicators.
Handles caching, cleaning, and alignment of daily time series.
"""

import logging
import time
from pathlib import Path

import pandas as pd
from fredapi import Fred

from src.config import CONFIG, PROJECT_ROOT, get_fred_api_key

logger = logging.getLogger(__name__)


def fetch_fred_series(
    fred: Fred,
    series_code: str,
    series_name: str,
    start_date: str,
    max_retries: int = 3,
) -> pd.Series:
    """
    Fetch a single series from the FRED API with retry logic.

    Args:
        fred: Authenticated Fred client.
        series_code: FRED series identifier.
        series_name: Human-readable name for logging.
        start_date: Start date in YYYY-MM-DD format.
        max_retries: Number of retry attempts on failure.

    Returns:
        pd.Series with DatetimeIndex.

    Raises:
        ValueError: If the fetched series is empty after all retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Fetching %s (%s)... [attempt %d/%d]",
                series_name, series_code, attempt, max_retries,
            )

            series = fred.get_series(series_code, observation_start=start_date)

            if series is None or series.empty:
                raise ValueError(f"Empty series for {series_name} ({series_code})")

            series.name = series_name
            series.index.name = "date"
            series.index = pd.to_datetime(series.index)

            n_obs = series.dropna().shape[0]
            logger.info(
                "  -> %d observations, %s to %s",
                n_obs,
                series.dropna().index.min().date(),
                series.dropna().index.max().date(),
            )

            return series

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "  Attempt %d failed: %s. Retrying in %ds...",
                    attempt, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("  All %d attempts failed for %s", max_retries, series_name)
                raise

def fetch_all_series(config: dict = CONFIG) -> pd.DataFrame:
    """
    Fetch all FRED series defined in config and assemble into a DataFrame.

    Args:
        config: Configuration dict (uses config["fred"] section).

    Returns:
        pd.DataFrame with DatetimeIndex, one column per series.
        Contains NaN for weekends/holidays (not yet cleaned).
    """
    api_key = get_fred_api_key()
    fred = Fred(api_key=api_key)

    series_map = config["fred"]["series"]
    start_date = config["fred"]["start_date"]

    all_series = {}
    for name, code in series_map.items():
        series = fetch_fred_series(fred, code, name, start_date)
        all_series[name] = series
        time.sleep(0.5)  # respect FRED rate limits (120 req/min)

    df = pd.concat(all_series.values(), axis=1)
    df.index.name = "date"
    df = df.sort_index()

    logger.info(
        "Assembled master DataFrame: %d rows x %d columns, %s to %s",
        len(df), len(df.columns),
        df.index.min().date(), df.index.max().date(),
    )

    return df


def save_raw_series(df: pd.DataFrame, config: dict = CONFIG) -> None:
    """Save each column as an individual CSV in data/raw/."""
    raw_dir = PROJECT_ROOT / config["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        path = raw_dir / f"{col}.csv"
        df[[col]].dropna().to_csv(path)
        logger.info("Saved raw series: %s", path)


def save_master(df: pd.DataFrame, config: dict = CONFIG) -> Path:
    """Save the master DataFrame to data/processed/master.csv."""
    proc_dir = PROJECT_ROOT / config["paths"]["processed"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    path = proc_dir / "master.csv"
    df.to_csv(path)
    logger.info("Saved master DataFrame: %s (%d rows)", path, len(df))

    return path


def load_master_if_cached(config: dict = CONFIG) -> pd.DataFrame | None:
    """
    Load master.csv from cache if it exists and was created today.

    Returns:
        pd.DataFrame if cache is fresh, None otherwise.
    """
    from datetime import datetime

    path = PROJECT_ROOT / config["paths"]["processed"] / "master.csv"

    if not path.exists():
        return None

    modified = datetime.fromtimestamp(path.stat().st_mtime)
    if modified.date() != datetime.today().date():
        logger.info("Cache exists but is stale (%s). Re-fetching.", modified.date())
        return None

    logger.info("Loading from cache: %s", path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def clean_master(df: pd.DataFrame, config: dict = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the raw master DataFrame.

    Steps:
        1. Save a trading_mask (True where data was originally observed)
        2. Forward-fill NaN (standard for daily financial series)
        3. Drop rows where ALL columns are NaN
        4. Run quality assertions

    Args:
        df: Raw DataFrame with NaN on weekends/holidays.

    Returns:
        Tuple of (cleaned DataFrame, trading_mask DataFrame).
        The trading_mask has the same shape as the cleaned DataFrame,
        with True for real trading days and False for forward-filled days.
        Save this mask — it's needed for correct volatility calculation.
    """
    logger.info("Cleaning master DataFrame...")
    logger.info("  NaN before cleaning:\n%s", df.isna().sum().to_string())

    # 1. Save trading mask BEFORE forward-fill
    trading_mask = df.notna()

    # 2. Forward-fill (max 5 days to avoid filling across long gaps)
    df = df.ffill(limit=5)

    # 3. Drop rows where everything is still NaN
    rows_before = len(df)
    df = df.dropna(how="all")
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info("  Dropped %d all-NaN rows", rows_dropped)

    # 4. Drop any remaining rows with NaN (start of series misalignment)
    df = df.dropna()

    # Align trading_mask to cleaned index
    trading_mask = trading_mask.loc[df.index]

    # 5. Quality assertions
    assert df.isna().sum().sum() == 0, "NaN remaining after cleaning"
    assert len(df) > 1000, f"Only {len(df)} rows — expected 5000+"

    # Check for negative spreads (would indicate data corruption)
    spread_cols = [c for c in df.columns if "spread" in c]
    for col in spread_cols:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            logger.warning("  %s has %d negative values — check data quality", col, n_neg)

    logger.info(
        "  Cleaned: %d rows x %d columns, %s to %s",
        len(df), len(df.columns),
        df.index.min().date(), df.index.max().date(),
    )

    return df, trading_mask





