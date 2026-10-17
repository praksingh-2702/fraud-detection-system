from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.data import load_paysim, time_split
from src.features.basic import FEATURES, build_features

MODEL_PATH = Path("models/baseline_lgbm.joblib")


def recall_at_fpr(y_true, scores, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[f1.argmax()])


def main() -> None:
    df = load_paysim()
    train, val, test = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part):,} rows, "
              f"{part['isFraud'].sum():,} fraud ({part['isFraud'].mean():.4%})")

    X_train, y_train = build_features(train), train["isFraud"]
    X_val, y_val = build_features(val), val["isFraud"]
    X_test, y_test = build_features(test), test["isFraud"]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        metric="average_precision",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    threshold = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])

    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print("\n=== TEST RESULTS ===")
    print(f"PR-AUC:                 {average_precision_score(y_test, scores):.4f}")
    print(f"Recall @ 1% FPR:        {recall_at_fpr(y_test, scores, 0.01):.4f}")
    print(f"Recall @ 0.1% FPR:      {recall_at_fpr(y_test, scores, 0.001):.4f}")
    print(f"Threshold (val F1-max): {threshold:.4f}")
    print(f"Precision: {tp / max(tp + fp, 1):.4f}   Recall: {tp / max(tp + fn, 1):.4f}")
    print(f"Confusion matrix -> TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop features:", imp[:6])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")


if __name__ == "__main__":
    main()