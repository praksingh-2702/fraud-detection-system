"""Receiver-history features: what the receiving account saw BEFORE this transaction."""
import pandas as pd

HISTORY_FEATURES = [
    "receiver_prior_count",
    "receiver_prior_amount",
    "receiver_is_new",
]


def add_receiver_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return history features with the same index as df.

    Only PRIOR transactions count. The current one is never included.
    """
    ordered = df.sort_values("step", kind="stable")
    grouped = ordered.groupby("nameDest", sort=False)

    out = pd.DataFrame(index=ordered.index)
    out["receiver_prior_count"] = grouped.cumcount().astype("int32")
    # running total minus the current amount = total of earlier transactions
    out["receiver_prior_amount"] = grouped["amount"].cumsum() - ordered["amount"]
    out["receiver_is_new"] = (out["receiver_prior_count"] == 0).astype("int8")
    return out.reindex(df.index)