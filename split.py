#!/usr/bin/env python
"""
Split the original telemetry into train and test sets for time-series analysis.
================================================================================
• train.csv          (the first N - n rows of the dataset)
• val_test.csv       (the last n rows of the dataset)

This split is standard for time-series forecasting, where you train on past
data and validate/test on the most recent data to simulate a real-world scenario.
"""

from pathlib import Path
import argparse
import pandas as pd

p = argparse.ArgumentParser(
    description="Split a time-series CSV into training (start) and testing (end) sets."
)
p.add_argument("--src",  default="ev9_SG.csv", help="Original telemetry file")
p.add_argument("--n",    type=int, default=2000,  help="Number of rows from the END to reserve for validation/testing")
p.add_argument("--outdir", default=".", help="Directory to write the new CSVs")
args = p.parse_args()

src     = Path(args.src)
outdir  = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

print(f"Reading source file: {src}")
df = pd.read_csv(src)

if len(df) <= args.n:
    raise ValueError(
        f"The source file only has {len(df)} rows, which is not enough to create a test set of size n={args.n}."
    )

# --- CHANGE: Slicing the DataFrame from the end ---
# The train set is everything EXCEPT the last `n` rows.
train    = df.iloc[:-args.n]

# The validation/test set IS the last `n` rows.
val_test = df.iloc[-args.n:]

# Define output filenames
train_out_path = outdir / "ev9_train.csv"
test_out_path = outdir / "ev9_test.csv"

# Save the new CSVs
val_test.to_csv(test_out_path, index=False)
train.to_csv(train_out_path, index=False)

print("\n✅  Successfully split the data:")
print(f"   Train set: {len(train):6d} rows (from the start) ➜ {train_out_path}")
print(f"   Test set:  {len(val_test):6d} rows (from the end)   ➜ {test_out_path}")