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



