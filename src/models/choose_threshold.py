"""Choose the alert cutoff (threshold) by cost and save it next to the model."""
import json

import lightgbm as lgb
import numpy as np

from src.data import load_paysim, time_split
from src.features.basic import build_features
from src.models.train_final import META_FILE, MODEL_FILE

# Cost model (amounts are in PaySim's own currency units).
# Every flagged transaction is reviewed by a person: REVIEW_COST each.
# Every fraud we do NOT flag costs its transaction amount.
REVIEW_COST = 10.0


def evaluate(scores, y, amount, threshold):
    flagged = scores >= threshold
    missed = (~flagged) & (y == 1)
    false_alarms = flagged & (y == 0)
    cost = float(amount[missed].sum() + REVIEW_COST * flagged.sum())
    return {
        "cost": cost,
        "flagged": int(flagged.sum()),
        "false_alarms": int(false_alarms.sum()),
        "missed_fraud": int(missed.sum()),
    }


def main() -> None:
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    df = load_paysim()
    _, val, test = time_split(df)

    val_scores = booster.predict(build_features(val))
    test_scores = booster.predict(build_features(test))
    y_val, a_val = val["isFraud"].to_numpy(), val["amount"].to_numpy()
    y_test, a_test = test["isFraud"].to_numpy(), test["amount"].to_numpy()

    # 1. Try many cutoffs on the validation period
    thresholds = np.round(np.arange(0.005, 1.0, 0.005), 3)
    costs = np.array(
        [evaluate(val_scores, y_val, a_val, t)["cost"] for t in thresholds]
    )

    # 2. Cutoffs within half a review fee of the cheapest count as tied.
    #    Take the one in the middle of that flat region.
    tied = thresholds[costs <= costs.min() + REVIEW_COST / 2]
    chosen = float(tied[np.abs(tied - np.median(tied)).argmin()])
    print(f"Cheapest cost on validation: {costs.min():,.0f}")
    print(f"Tied cutoffs: {tied.min()} to {tied.max()}  ->  chosen: {chosen}")

    # 3. Check on the test period (not used for choosing)
    print("\nTEST PERIOD (review fee = %s):" % REVIEW_COST)
    rows = [
        ("No model (flag nothing)", 2.0),
        ("Flag everything", -1.0),
        ("Cutoff 0.5", 0.5),
        (f"Chosen cutoff {chosen}", chosen),
    ]
    results = {}
    for name, t in rows:
        r = evaluate(test_scores, y_test, a_test, t)
        results[name] = r
        print(f"  {name:<26} cost={r['cost']:>16,.0f}  flagged={r['flagged']:>7,}  "
              f"false alarms={r['false_alarms']:>6,}  missed fraud={r['missed_fraud']:>5,}")

    no_model = results["No model (flag nothing)"]["cost"]
    ours = results[f"Chosen cutoff {chosen}"]["cost"]
    print(f"\nCost reduction vs no model: {1 - ours / no_model:.2%}")

    # 4. Save the cutoff next to the model so the API can load it
    meta = json.loads(META_FILE.read_text())
    meta["threshold"] = chosen
    meta["threshold_plateau"] = [float(tied.min()), float(tied.max())]
    meta["cost_model"] = {
        "review_cost": REVIEW_COST,
        "missed_fraud_cost": "transaction amount",
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"\nUpdated {META_FILE}")


if __name__ == "__main__":
    main()