"""
Historical stress testing engine.

Identifies stress episodes from market conditions and measures
the forward impact on credit spreads across multiple horizons.
"""

import logging
import pandas as pd

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
    Measure spread impact at fixed horizons after each trigger date.

    First version: point-to-point impact only (final - initial).

    Args:
        df: DataFrame with spread data.
        trigger_dates: List of stress event dates.
        spread_col: Spread column to measure.
        forward_windows: List of forward horizons in trading days.

    Returns:
        DataFrame with one row per episode, columns for each horizon.
    """
    records = []

    for trigger in trigger_dates:
        if trigger not in df.index:
            continue

        trigger_loc = df.index.get_loc(trigger)
        spread_at_trigger = df[spread_col].iloc[trigger_loc]
        record = {
            "trigger_date": trigger,
            "spread_at_trigger": spread_at_trigger,
        }

        for window in forward_windows:
            end_loc = trigger_loc + window
            if end_loc >= len(df):
                record[f"impact_{window}d"] = np.nan
                continue

            spread_at_end = df[spread_col].iloc[end_loc]
            record[f"impact_{window}d"] = spread_at_end - spread_at_trigger

        records.append(record)

    return pd.DataFrame(records)




