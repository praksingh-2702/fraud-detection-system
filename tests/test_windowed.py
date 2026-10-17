import pandas as pd

from src.features.windowed import add_windowed_history


def test_window_counts_only_recent_prior_transactions():
    df = pd.DataFrame({
        "step": [1, 5, 30, 31],
        "nameDest": ["A", "A", "A", "A"],
        "amount": [10.0, 20.0, 40.0, 80.0],
    })
    h = add_windowed_history(df, windows=(24,))
    # step 30 looks back to step 6 -> sees neither step 1 nor 5
    # step 31 looks back to step 7 -> sees only step 30
    assert list(h["receiver_count_24h"]) == [0, 1, 0, 1]
    assert list(h["receiver_amount_24h"]) == [0.0, 10.0, 0.0, 40.0]


def test_accounts_do_not_mix():
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "nameDest": ["A", "B", "A"],
        "amount": [10.0, 99.0, 5.0],
    })
    h = add_windowed_history(df, windows=(24,))
    assert list(h["receiver_count_24h"]) == [0, 0, 1]
    assert list(h["receiver_amount_24h"]) == [0.0, 0.0, 10.0]