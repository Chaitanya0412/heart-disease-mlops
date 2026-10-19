
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_data():
    """Download UCI Heart Disease data and save a local CSV."""
    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    df = X.copy()
    df["target"] = pd.to_numeric(y, errors="coerce")

    # Handle missing values represented by '?' or invalid entries.
    df = df.replace("?", pd.NA)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(DATA_DIR / "heart.csv", index=False)
    print(f"Saved {len(df)} rows to data/heart.csv")
    return df


def load_data():
    """Load the downloaded dataset."""
    path = DATA_DIR / "heart.csv"

    if not path.exists():
        return download_data()

    return pd.read_csv(path)


if __name__ == "__main__":
    download_data()
