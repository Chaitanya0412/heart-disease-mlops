
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_data

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_data()

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First Five Rows ---")
    print(df.head())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isna().sum())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Target Balance ---")
    print(df["target"].value_counts())

    # Chart 1: Target class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close()

    # Chart 2: Histograms
    df.hist(figsize=(14, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "histograms.png", dpi=150)
    plt.close()

    # Chart 3: Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation.png", dpi=150)
    plt.close()

    print("\nEDA completed.")
    print("Charts saved in reports/figures/")


if __name__ == "__main__":
    main()
