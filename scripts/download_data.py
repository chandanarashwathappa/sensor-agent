"""Re-download the AI4I 2020 dataset from the UCI repository (optional).

    pip install ucimlrepo
    python scripts/download_data.py
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

out = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"
ds = fetch_ucirepo(id=601)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"saved {df.shape[0]} rows to {out}")
print(df["Machine failure"].value_counts())
