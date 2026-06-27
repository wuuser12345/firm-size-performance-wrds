from pathlib import Path

import numpy as np
import pandas as pd


raw_file = Path("data/raw/wrds_manual_pull.csv")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

if not raw_file.exists():
    raise FileNotFoundError(
        "Manual WRDS file not found. Please save it as data/raw/wrds_manual_pull.csv"
    )

print(f"Reading manual WRDS file: {raw_file}")

df = pd.read_csv(raw_file)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

print("Columns found:")
print(df.columns.tolist())

# Keep relevant variables if available
keep_cols = [
    "gvkey",
    "conm",
    "fic",
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

# Create analysis variables
df["roa"] = df["nicon"] / df["at"]
df["firm_size"] = np.log(df["at"])
df["leverage"] = (df["dltt"] + df["dlc"]) / df["seq"]

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Basic filter
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
    f.write("Data source: Manual WRDS web download\n")
    f.write(f"Input file: {raw_file}\n")
    f.write(f"Clean rows: {len(df)}\n")
    f.write(f"Columns: {len(df.columns)}\n")
    f.write("\nColumns in clean panel:\n")
    for col in df.columns:
        f.write(f"- {col}\n")

print("Cleaning completed successfully.")
print(f"Clean panel saved to: {output_file}")
print(f"Cleaning log saved to: {clean_log}")
print(df.head())
