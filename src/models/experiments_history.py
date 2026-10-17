"""Does adding receiver-history features help? Same split, same settings."""
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import load_paysim, time_split
from src.features.basic import build_features
from src.features.history import HISTORY_FEATURES, add_receiver_history
from src.models.train_baseline import recall_at_fpr
from src.models.train_final import PARAMS

# Receivers of the 4 frauds the current model missed
MISSED_RECEIVERS = ["C1916793179", "C1770418982", "C1909486199", "C766725973"]


def main() -> None:
    df = load_paysim()
    history = add_receiver_history(df)  # computed on the FULL timeline
    parts = dict(zip(["train", "val", "test"], time_split(df)))

    for label, with_history in [("current features", False),
                                ("+ receiver history", True)]:
        X = {}
        for name, part in parts.items():
            feats = build_features(part)
            if with_history:
                feats = pd.concat(
                    [feats, history.loc[part.index, HISTORY_FEATURES]], axis=1
                )
            X[name] = feats

        dtrain = lgb.Dataset(X["train"], label=parts["train"]["isFraud"])
        dval = lgb.Dataset(X["val"], label=parts["val"]["isFraud"], reference=dtrain)
        booster = lgb.train(
            PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = booster.best_iteration

        yt = parts["test"]["isFraud"].to_numpy()
        ts = booster.predict(X["test"], num_iteration=best)
        print(f"\n=== {label} ===")
        print(f"trees used: {best}")
        print(f"test PR-AUC: {average_precision_score(yt, ts):.4f} | "
              f"recall@0.01%FPR: {recall_at_fpr(yt, ts, 0.0001):.4f}")
        print(f"at cutoff 0.5: missed frauds = {int(((ts < 0.5) & (yt == 1)).sum())}, "
              f"false alarms = {int(((ts >= 0.5) & (yt == 0)).sum())}")

        print("scores of the 4 frauds missed earlier:")
        for name in ("val", "test"):
            part = parts[name]
            scores = booster.predict(X[name], num_iteration=best)
            mask = (part["nameDest"].isin(MISSED_RECEIVERS)
                    & (part["isFraud"] == 1)).to_numpy()
            rows = part.loc[mask, ["nameDest", "amount"]].itertuples(index=False)
            for (dest, amount), score in zip(rows, scores[mask]):
                print(f"  {name:<5} {dest}  amount={amount:>12,.0f}  score={score:.4f}")


if __name__ == "__main__":
    main()