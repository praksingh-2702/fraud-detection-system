import pandas as pd

TYPE_CODES = {"PAYMENT": 0, "CASH_IN": 1, "CASH_OUT": 2, "DEBIT": 3, "TRANSFER": 4}

FEATURES = [
    "amount",
    "type_code",
    "hour_of_day",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "orig_drained",
    "amount_to_balance",
    "dest_is_merchant",
    "dest_empty_before",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless per-transaction features (velocity features come later)."""
    X = pd.DataFrame(index=df.index)
    X["amount"] = df["amount"]
    X["type_code"] = df["type"].map(TYPE_CODES).astype("int8")
    X["hour_of_day"] = ((df["step"] - 1) % 24).astype("int8")
    X["oldbalanceOrg"] = df["oldbalanceOrg"]
    X["newbalanceOrig"] = df["newbalanceOrig"]
    X["oldbalanceDest"] = df["oldbalanceDest"]
    X["newbalanceDest"] = df["newbalanceDest"]
    # Both are 0 when the books balance; fraud rows often break this.
    X["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    X["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    X["orig_drained"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype("int8")
    X["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    X["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype("int8")
    X["dest_empty_before"] = (df["oldbalanceDest"] == 0).astype("int8")
    return X[FEATURES]