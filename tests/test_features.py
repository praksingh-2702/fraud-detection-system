import pandas as pd

from src.features.basic import FEATURES, build_features


def _rows():
    return pd.DataFrame({
        "step": [1, 25],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 181.0],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [500.0, 181.0],
        "newbalanceOrig": [400.0, 0.0],
        "nameDest": ["M1", "C3"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 0.0],
    })


def test_feature_columns_and_order():
    assert list(build_features(_rows()).columns) == FEATURES


def test_balanced_payment_has_zero_error():
    X = build_features(_rows())
    assert X.loc[0, "orig_balance_error"] == 0
    assert X.loc[0, "dest_is_merchant"] == 1
    assert X.loc[0, "orig_drained"] == 0


def test_drained_transfer_flagged():
    X = build_features(_rows())
    assert X.loc[1, "orig_drained"] == 1
    assert X.loc[1, "dest_balance_error"] == 181.0
    assert X.loc[1, "hour_of_day"] == 0  # step 25 is hour 0 of day 2