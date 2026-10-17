from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/paysim.csv")
CACHE_PATH = Path("data/paysim.parquet")

DTYPES = {
    "step": "int16",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_paysim(path: Path = RAW_PATH, use_cache: bool = True) -> pd.DataFrame:
    """Load PaySim. The first call reads the CSV and caches a parquet copy."""
    if use_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    df = pd.read_csv(path, dtype=DTYPES)
    # isFlaggedFraud is the simulator's own rule-based flag, so we don't use it.
    df = df.drop(columns=["isFlaggedFraud"])
    if use_cache:
        df.to_parquet(CACHE_PATH, index=False)
    return df


def time_split(df: pd.DataFrame, train_end: int = 520, val_end: int = 630):
    """Split by time (step = hour): train <= 520 < val <= 630 < test.

    A random split would leak future patterns into training.
    """
    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return train, val, test