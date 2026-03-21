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


def compute_zscore(series,window):
    
    rolling_mean = series.rolling( window , min_periods = window // 2 ).mean()
    rolling_std = series.rolling( window , min_periods = window // 2 ).std()

    zscore = np.where( rolling_std != 0, (series - rolling_mean) / rolling_std , 0.0)

    return pd.Series(zscore, index=series.index)


