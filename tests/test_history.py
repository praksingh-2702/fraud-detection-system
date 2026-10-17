import pandas as pd

from src.features.history import add_receiver_history


def test_counts_only_prior_transactions():
    df = pd.DataFrame({
        "step": [1, 1, 2, 3],
        "nameDest": ["A", "B", "A", "A"],
        "amount": [10.0, 5.0, 20.0, 1.0],
    })
    h = add_receiver_history(df)
    assert list(h["receiver_prior_count"]) == [0, 0, 1, 2]
    assert list(h["receiver_prior_amount"]) == [0.0, 0.0, 10.0, 30.0]
    assert list(h["receiver_is_new"]) == [1, 1, 0, 0]