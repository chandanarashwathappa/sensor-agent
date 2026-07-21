from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=601)

df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
df.to_csv("data/ai4i2020.csv", index=False)

print("Shape (rows, columns):", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nHow many machines failed:")
print(df["Machine failure"].value_counts())