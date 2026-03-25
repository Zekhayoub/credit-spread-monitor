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



