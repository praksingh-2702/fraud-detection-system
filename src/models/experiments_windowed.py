"""Compare feature sets: current, all-time history, and recent-window history."""
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import load_paysim, time_split
from src.features.basic import build_features
from src.features.history import add_receiver_history
from src.features.windowed import WINDOW_FEATURES, add_windowed_history
from src.models.train_baseline import recall_at_fpr
from src.models.train_final import PARAMS

# Receivers of the 4 frauds the current model missed
MISSED_RECEIVERS = ["C1916793179", "C1770418982", "C1909486199", "C766725973"]


def main() -> None:
    df = load_paysim()
    old = add_receiver_history(df)       # all-time history (the version that hurt)
    recent = add_windowed_history(df)    # last 24 hours / last 7 days
    extra = pd.concat([old[["receiver_is_new"]], recent], axis=1)
    parts = dict(zip(["train", "val", "test"], time_split(df)))

    # Does the "is this receiver new" signal drift over time?
    print("Share of transfers/cash-outs going to a NEW receiver, by period:")
    for name, part in parts.items():
        risky = part[part["type"].isin(["TRANSFER", "CASH_OUT"])]
        share = extra.loc[risky.index, "receiver_is_new"].mean()
        print(f"  {name:<6} {share:.1%}")

    variants = {
        "current features": [],
        "+ is_new only": ["receiver_is_new"],
        "+ recent windows": list(WINDOW_FEATURES),
        "+ is_new + windows": ["receiver_is_new"] + list(WINDOW_FEATURES),
    }

    rows = []
    for label, cols in variants.items():
        print(f"training: {label} ...", flush=True)
        X = {}
        for name, part in parts.items():
            feats = build_features(part)
            if cols:
                feats = pd.concat([feats, extra.loc[part.index, cols]], axis=1)
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

        rescued = 0
        for name in ("val", "test"):
            part = parts[name]
            sc = booster.predict(X[name], num_iteration=best)
            mask = (part["nameDest"].isin(MISSED_RECEIVERS)
                    & (part["isFraud"] == 1)).to_numpy()
            rescued += int((sc[mask] >= 0.5).sum())

        rows.append({
            "features": label,
            "trees": best,
            "test_PR_AUC": round(average_precision_score(yt, ts), 4),
            "rec@0.01%FPR": round(recall_at_fpr(yt, ts, 0.0001), 4),
            "missed@0.5": int(((ts < 0.5) & (yt == 1)).sum()),
            "false_alarms@0.5": int(((ts >= 0.5) & (yt == 0)).sum()),
            "old_4_misses_now_caught": rescued,
        })

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()