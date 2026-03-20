"""
FRED API data ingestion.

Downloads credit spread series, Treasury yields, and market indicators.
Handles caching, cleaning, and alignment of daily time series.
"""

import logging

import pandas as pd
from fredapi import Fred

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