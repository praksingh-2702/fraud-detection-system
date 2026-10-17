"""Quick check: do receiver accounts have history, and does fraud look different?"""
from src.data import load_paysim

MISSED_RECEIVERS = [
    "C1916793179", "C1770418982", "C1909486199", "C766725973",
]


def main() -> None:
    df = load_paysim()
    # Oldest first, then count how many earlier transactions
    # each receiver account had at the moment of this one.
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df["prior_to_receiver"] = df.groupby("nameDest").cumcount()

    risky = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    print("Earlier transactions to the same receiver (TRANSFER and CASH_OUT only):")
    print(risky.groupby("isFraud")["prior_to_receiver"].describe().to_string())
    print("\nShare of receivers with at least 1 earlier transaction:")
    print(risky.groupby("isFraud")["prior_to_receiver"].apply(lambda s: (s > 0).mean()))

    print("\nFull history of the receivers in the 4 missed frauds:")
    cols = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud", "prior_to_receiver"]
    print(df[df["nameDest"].isin(MISSED_RECEIVERS)][cols]
          .sort_values(["nameDest", "step"]).to_string(index=False))


if __name__ == "__main__":
    main()