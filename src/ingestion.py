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
) -> pd.Series:
    """
    Fetch a single series from the FRED API.

    Args:
        fred: Authenticated Fred client.
        series_code: FRED series identifier (e.g., "BAMLC0A4CBBB").
        series_name: Human-readable name for logging (e.g., "bbb_spread").
        start_date: Start date in YYYY-MM-DD format.

    Returns:
        pd.Series with DatetimeIndex and float values.

    Raises:
        ValueError: If the fetched series is empty.
    """
    logger.info("Fetching %s (%s) from %s...", series_name, series_code, start_date)

    try:
        series = fred.get_series(series_code, observation_start=start_date)
    except Exception as e:
        logger.error("Failed to fetch %s (%s): %s", series_name, series_code, e)
        raise

    if series is None or series.empty:
        raise ValueError(f"Empty series returned for {series_name} ({series_code})")

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