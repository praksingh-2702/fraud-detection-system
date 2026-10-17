"""Train the final fraud model and record the run in MLflow.

Setup: plain LightGBM, no class weighting (config C from the experiments).
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.metrics import average_precision_score

from src.data import load_paysim, time_split
from src.features.basic import FEATURES, build_features
from src.models.train_baseline import recall_at_fpr

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "fraud_lgbm.txt"
META_FILE = MODEL_DIR / "fraud_lgbm_meta.json"

PARAMS = {
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


def main() -> None:
    # 1. Load data and split by time (same defaults as before: 520 / 630)
    df = load_paysim()
    train, val, test = time_split(df)
    X_train, y_train = build_features(train), train["isFraud"]
    X_val, y_val = build_features(val), val["isFraud"]
    X_test, y_test = build_features(test), test["isFraud"]

    # 2. Train, stopping when the validation score stops improving
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best = booster.best_iteration

    # 3. Score the validation and test periods
    val_scores = booster.predict(X_val, num_iteration=best)
    test_scores = booster.predict(X_test, num_iteration=best)
    yt = y_test.to_numpy()
    metrics = {
        "best_iteration": best,
        "val_pr_auc": round(float(average_precision_score(y_val, val_scores)), 4),
        "test_pr_auc": round(float(average_precision_score(yt, test_scores)), 4),
        "test_recall_at_0_1pct_fpr": round(recall_at_fpr(yt, test_scores, 0.001), 4),
        "test_recall_at_0_01pct_fpr": round(recall_at_fpr(yt, test_scores, 0.0001), 4),
    }

    # 4. Save the model (only the trees up to the best iteration) and its details
    MODEL_DIR.mkdir(exist_ok=True)
    booster.save_model(str(MODEL_FILE), num_iteration=best)
    meta = {
        "features": FEATURES,
        "params": PARAMS,
        "split": {"train_end_step": 520, "val_end_step": 630},
        "metrics": metrics,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    # 5. Safety check: does the saved file behave like the trained model?
    reloaded = lgb.Booster(model_file=str(MODEL_FILE))
    sample = X_test.head(1000)
    same = np.allclose(
        booster.predict(sample, num_iteration=best), reloaded.predict(sample)
    )
    print("Reload check:", "OK" if same else "MISMATCH")

    # 6. Record the run in MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="final_no_weight"):
        mlflow.log_params(
            {**PARAMS, "n_features": len(FEATURES),
             "train_end_step": 520, "val_end_step": 630}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(META_FILE))

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print(f"\nSaved {MODEL_FILE} and {META_FILE}")


if __name__ == "__main__":
    main()