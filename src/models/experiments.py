"""Compare imbalance handling and feature choices on the time-based split."""
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import load_paysim, time_split
from src.features.basic import FEATURES, build_features
from src.models.train_baseline import best_f1_threshold, recall_at_fpr

BASE = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "seed": 42,
    "verbosity": -1,
}


def run(name, feats, params, X, y):
    dtrain = lgb.Dataset(X["train"][feats], label=y["train"])
    dval = lgb.Dataset(X["val"][feats], label=y["val"], reference=dtrain)
    history = {}
    booster = lgb.train(
        {**BASE, **params},
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.record_evaluation(history),
        ],
    )
    best = booster.best_iteration
    curve = history["val"]["average_precision"]

    val_scores = booster.predict(X["val"][feats], num_iteration=best)
    test_scores = booster.predict(X["test"][feats], num_iteration=best)
    thr = best_f1_threshold(y["val"], val_scores)
    pred = test_scores >= thr
    yt = y["test"].to_numpy()

    return {
        "config": name,
        "best_iter": best,
        "val_AP@best": round(curve[best - 1], 4),
        "val_AP@last": round(curve[-1], 4),
        "test_AP": round(average_precision_score(yt, test_scores), 4),
        "rec@0.1%FPR": round(recall_at_fpr(yt, test_scores, 0.001), 4),
        "rec@0.01%FPR": round(recall_at_fpr(yt, test_scores, 0.0001), 4),
        "FP": int((pred & (yt == 0)).sum()),
        "FN": int((~pred & (yt == 1)).sum()),
    }


def main() -> None:
    df = load_paysim()
    parts = dict(zip(["train", "val", "test"], time_split(df)))
    X = {k: build_features(p) for k, p in parts.items()}
    y = {k: p["isFraud"] for k, p in parts.items()}

    pos_w = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    no_hour = [f for f in FEATURES if f != "hour_of_day"]
    reg = {"num_leaves": 15, "min_data_in_leaf": 200, "lambda_l2": 10.0}

    configs = [
        ("A_full_weight",     FEATURES, {"scale_pos_weight": pos_w}),
        ("B_sqrt_weight",     FEATURES, {"scale_pos_weight": pos_w ** 0.5}),
        ("C_no_weight",       FEATURES, {}),
        ("D_no_weight_no_hr", no_hour,  {}),
        ("E_sqrt_reg_no_hr",  no_hour,  {"scale_pos_weight": pos_w ** 0.5, **reg}),
    ]

    rows = []
    for name, feats, params in configs:
        print(f"running {name} ...", flush=True)
        rows.append(run(name, feats, params, X, y))

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
    