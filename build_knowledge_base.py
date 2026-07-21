import pandas as pd
import os

# Load the dataset we saved earlier
df = pd.read_csv("data/ai4i2020.csv")

# We'll collect text documents in this list
documents = []

# --- DOCUMENT 1: Data dictionary (domain knowledge) ---
data_dictionary = """
DATA DICTIONARY - AI4I 2020 Predictive Maintenance Dataset

This dataset contains 10,000 records from industrial milling machines.

SENSOR AND PROCESS PARAMETERS:
- Type: Product quality variant. L = low quality (60% of products), M = medium (30%), H = high (10%).
- Air temperature: Ambient air temperature around the machine, in Kelvin (K). Typically around 300 K.
- Process temperature: Temperature of the machining process, in Kelvin (K). Normally about 10 K higher than air temperature.
- Rotational speed: Spindle rotation speed in revolutions per minute (rpm), derived from a power of 2860 W.
- Torque: Torque applied during machining, in Newton-metres (Nm). Usually around 40 Nm.
- Tool wear: Cumulative minutes of tool usage. Higher quality products add more wear time.

TARGET:
- Machine failure: 1 if the machine failed, 0 if it operated normally. A machine fails if any of the five failure modes below occurs.
"""
documents.append(data_dictionary)

# --- DOCUMENT 2: Failure mode descriptions (domain knowledge) ---
failure_modes = """
FAILURE MODES - AI4I 2020 Predictive Maintenance Dataset

There are five distinct failure modes. A machine is marked as failed if any one occurs.

1. TWF (Tool Wear Failure): The tool wears out and fails. Occurs when tool wear reaches a high randomized threshold (around 200-240 minutes of use). The tool must be replaced or the process stops.

2. HDF (Heat Dissipation Failure): Heat cannot be dissipated fast enough. Occurs when the difference between air and process temperature is below about 8.6 K, combined with a low rotational speed (below roughly 1380 rpm).

3. PWF (Power Failure): The power required for the process is outside the acceptable range. Power is torque multiplied by rotational speed (in rad/s). Failure occurs when this power is below about 3500 W or above about 9000 W.

4. OSF (Overstrain Failure): The machine is overstrained. Occurs when the product of tool wear and torque exceeds a threshold that depends on product quality (around 11000 minNm for L, higher for M and H).

5. RNF (Random Failure): A random failure with no specific cause. Each machine has a very small (about 0.1%) chance of failing randomly regardless of conditions.
"""
documents.append(failure_modes)

# --- DOCUMENT 3: Computed statistics from the real data ---
total = len(df)
failures = df["Machine failure"].sum()
fail_rate = failures / total * 100

stats = f"""
DATASET STATISTICS - Computed from the actual AI4I 2020 data

- Total machine records: {total}
- Total failures: {failures} ({fail_rate:.1f}% failure rate)
- Normal operations: {total - failures}

FAILURE MODE COUNTS:
- TWF (Tool Wear Failure): {df['TWF'].sum()} occurrences
- HDF (Heat Dissipation Failure): {df['HDF'].sum()} occurrences
- PWF (Power Failure): {df['PWF'].sum()} occurrences
- OSF (Overstrain Failure): {df['OSF'].sum()} occurrences
- RNF (Random Failure): {df['RNF'].sum()} occurrences

AVERAGE SENSOR READINGS:
When machines operated NORMALLY (no failure):
- Air temperature: {df[df['Machine failure']==0]['Air temperature'].mean():.1f} K
- Process temperature: {df[df['Machine failure']==0]['Process temperature'].mean():.1f} K
- Rotational speed: {df[df['Machine failure']==0]['Rotational speed'].mean():.0f} rpm
- Torque: {df[df['Machine failure']==0]['Torque'].mean():.1f} Nm
- Tool wear: {df[df['Machine failure']==0]['Tool wear'].mean():.1f} minutes

When machines FAILED:
- Air temperature: {df[df['Machine failure']==1]['Air temperature'].mean():.1f} K
- Process temperature: {df[df['Machine failure']==1]['Process temperature'].mean():.1f} K
- Rotational speed: {df[df['Machine failure']==1]['Rotational speed'].mean():.0f} rpm
- Torque: {df[df['Machine failure']==1]['Torque'].mean():.1f} Nm
- Tool wear: {df[df['Machine failure']==1]['Tool wear'].mean():.1f} minutes
"""
documents.append(stats)

# --- Save all documents to a folder ---
os.makedirs("knowledge_base", exist_ok=True)
for i, doc in enumerate(documents):
    with open(f"knowledge_base/doc_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(doc)

print(f"Created {len(documents)} knowledge base documents in the 'knowledge_base' folder.")
print("\nPreview of the computed statistics:")
print(stats)