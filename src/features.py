import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

SPREAD_COLS = ["aaa_spread", "aa_spread", "bbb_spread", "hy_spread"]


def compute_spread_changes(df,spread_col,windows):
    if spread_col not in df.columns:
        raise KeyError(f"Column {spread_col} not found")

    result = pd.DataFrame(index=df.index)
    for w in windows:
        col_name = f"{spread_col}_change_{w}d"
        result[col_name] = df[spread_col].diff(w)

    return result


def compute_zscore(series,window,min_std):
    
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window, min_periods=window // 2).std()

    zscore = np.where( rolling_std > min_std,(series - rolling_mean) / rolling_std, 0.0)

    result = pd.Series(zscore, index=series.index)

    assert not np.isinf(result).any(), "Inf values in z-score — check min_std threshold"

    return result



def compute_percentile( series, window):

    # Vectorized, uses compiled C under the hood
    # faster than .rolling().apply(lambda x: ...)
    pctile = series.rolling(window, min_periods=window // 2).rank() / window

    return pctile



def compute_rolling_volatility( spread_changes, window, trading_mask ):

    changes = spread_changes.copy()

    if trading_mask is not None:
        # Set non-trading day changes to NaN so they don't contribute
        # zero-change observations that artificially compress volatility
        changes = changes.where(trading_mask)
        logger.debug("Volatility: excluded %d non-trading days", (~trading_mask).sum())

    vol = changes.rolling(window, min_periods=window // 2).std()
    return vol


def compute_compression_ratio(df ):

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




