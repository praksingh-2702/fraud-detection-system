"""Windowed receiver-history features: activity in the last N hours only."""
import numpy as np
import pandas as pd

WINDOWS = (24, 168)  # one day and one week (PaySim step = 1 hour)

WINDOW_FEATURES = (
    [f"receiver_count_{w}h" for w in WINDOWS]
    + [f"receiver_amount_{w}h" for w in WINDOWS]
)

_BIG = 10_000  # larger than any step + window, keeps accounts apart in the sort key


def add_windowed_history(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """For each transaction: how many transfers (and how much money) the receiving
    account got in the previous `w` hours. Only earlier transactions count.
    """
    codes = pd.factorize(df["nameDest"])[0].astype(np.int64)
    step = df["step"].to_numpy().astype(np.int64)
    amount = df["amount"].to_numpy().astype(np.float64)

    # Sort by account, then time (stable, so ties keep the original order)
    order = np.lexsort((step, codes))
    g, s, a = codes[order], step[order], amount[order]
    key = g * _BIG + s
    pos = np.arange(len(df))
    csum = np.concatenate(([0.0], np.cumsum(a)))  # csum[k] = money in rows before k

    out = {}
    for w in windows:
        # first row of the same account whose step is within the last w hours
        start = np.searchsorted(key, g * _BIG + (s - w), side="left")
        count = np.empty(len(df), dtype=np.int32)
        money = np.empty(len(df), dtype=np.float64)
        count[order] = pos - start
        money[order] = csum[pos] - csum[start]
        out[f"receiver_count_{w}h"] = count
        out[f"receiver_amount_{w}h"] = money
    return pd.DataFrame(out, index=df.index)[
        [f"receiver_count_{w}h" for w in windows]
        + [f"receiver_amount_{w}h" for w in windows]
    ]