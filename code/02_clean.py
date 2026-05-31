from pathlib import Path

import numpy as np
import pandas as pd


raw_base = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)


# Find most recent WRDS pull folder
pull_folders = [p for p in raw_base.iterdir() if p.is_dir()]

if not pull_folders:
    raise FileNotFoundError(
        "No WRDS pull folder found in data/raw/. "
        "Run code/01_pull_data.py first."
    )

latest_pull = max(pull_folders, key=lambda p: p.stat().st_mtime)
print(f"Using latest pull folder: {latest_pull}")


# Read all yearly parquet files
files = sorted(latest_pull.glob("fyear_*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No fyear_*.parquet files found in {latest_pull}."
    )

dfs = []
for file in files:
    print(f"Reading {file}")
    dfs.append(pd.read_parquet(file))

df = pd.concat(dfs, ignore_index=True)


# Standardize column names
df.columns = df.columns.str.lower()


# Keep only relevant variables
keep_cols = [
    "gvkey",
    "conm",
    "fic",
    "loc",
    "datadate",
    "fyear",
    "at",
    "nicon",
    "dlc",
    "dltt",
    "seq",
    "sic",
    "naics",
]

available_cols = [col for col in keep_cols if col in df.columns]
df = df[available_cols].copy()


# Convert numeric columns
numeric_cols = ["fyear", "at", "nicon", "dlc", "dltt", "seq"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Drop rows without firm id or fiscal year
df = df.dropna(subset=["gvkey", "fyear"])


# Drop duplicates
df = df.drop_duplicates(subset=["gvkey", "fyear"])


# Create variables for analysis
df["roa"] = df["nicon"] / df["at"]
df["firm_size"] = np.log(df["at"])
df["leverage"] = (df["dltt"] + df["dlc"]) / df["seq"]


# Replace impossible infinite values
df = df.replace([np.inf, -np.inf], np.nan)


# Basic filters
df = df[df["at"] > 0]


# Sort panel
df = df.sort_values(["gvkey", "fyear"])


# Save cleaned panel
output_file = processed_dir / "panel_clean.parquet"
df.to_parquet(output_file, index=False)


# Save cleaning log
clean_log = processed_dir / "clean_log.txt"

with open(clean_log, "w") as f:
    f.write("Cleaning log\n")
    f.write(f"Input folder: {latest_pull}\n")
    f.write(f"Raw rows after concat: {len(pd.concat(dfs, ignore_index=True))}\n")
    f.write(f"Clean rows: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n")
    f.write("\nColumns in clean panel:\n")
    for col in df.columns:
        f.write(f"- {col}\n")

print("Cleaning completed successfully.")
print(f"Clean panel saved to: {output_file}")
print(f"Cleaning log saved to: {clean_log}")