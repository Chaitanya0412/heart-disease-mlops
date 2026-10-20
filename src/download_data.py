
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

os.makedirs("data", exist_ok=True)

print("Downloading Heart Disease dataset...")

heart = fetch_ucirepo(id=45)

X = heart.data.features.copy()
y = heart.data.targets.iloc[:, 0]

df = X.copy()
df["target"] = y

# Remove rows where the target is missing
df = df.dropna(subset=["target"])

# Convert target to binary: 0 = absence, 1 = presence
df["target"] = (df["target"] > 0).astype(int)

df.to_csv("data/heart.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
