"""List the frauds the model misses, to see what they have in common."""
import lightgbm as lgb

from src.data import load_paysim, time_split
from src.features.basic import build_features
from src.models.train_final import MODEL_FILE


def main() -> None:
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    df = load_paysim()
    _, val, test = time_split(df)
    for name, part in [("VALIDATION", val), ("TEST", test)]:
        part = part.assign(score=booster.predict(build_features(part)))
        missed = part[(part["isFraud"] == 1) & (part["score"] < 0.5)]
        total = int(part["isFraud"].sum())
        print(f"\n{name}: {len(missed)} missed out of {total} frauds, "
              f"total amount missed = {missed['amount'].sum():,.0f}")
        print(missed.head(15).to_string())


if __name__ == "__main__":
    main()