from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import wrds
from dotenv import load_dotenv


# Load WRDS username from .env
load_dotenv()
WRDS_USERNAME = os.getenv("WRDS_USERNAME")

if WRDS_USERNAME is None:
    raise ValueError("WRDS_USERNAME is missing. Please check your .env file.")


# Create timestamped output folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
raw_dir = Path("data/raw") / timestamp
raw_dir.mkdir(parents=True, exist_ok=True)


# Variables for this research note
variables = [
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


print("Connecting to WRDS...")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

row_counts = {}

for year in range(2015, 2025):
    print(f"Pulling data for fiscal year {year}...")

    query = f"""
        SELECT {", ".join(variables)}
        FROM comp_global_daily.g_funda
        WHERE fyear = {year}
    """

    df = db.raw_sql(query)

    output_file = raw_dir / f"fyear_{year}.parquet"
    df.to_parquet(output_file, index=False)

    row_counts[year] = len(df)
    print(f"Saved {len(df)} rows to {output_file}")


# Save column schema
schema = pd.DataFrame(
    {
        "column": variables,
        "description": [
            "Global Company Key",
            "Company Name",
            "Country of Incorporation",
            "Country of Headquarters",
            "Data Date",
            "Fiscal Year",
            "Assets - Total",
            "Net Income (Loss) - Consolidated",
            "Debt in Current Liabilities - Total",
            "Long-Term Debt - Total",
            "Stockholders Equity - Parent",
            "Standard Industry Classification Code",
            "North American Industry Classification Code",
        ],
    }
)

schema.to_csv(raw_dir / "column_schema.csv", index=False)


# Save metadata
with open(raw_dir / "pull_metadata.txt", "w") as f:
    f.write("WRDS / Compustat Global Daily - Fundamentals Annual Pull\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write("Table: comp_global_daily.g_funda\n")
    f.write("Fiscal years: 2015-2024\n\n")
    f.write("Variables:\n")
    for var in variables:
        f.write(f"- {var}\n")

    f.write("\nRow counts by fiscal year:\n")
    for year, count in row_counts.items():
        f.write(f"{year}: {count} rows\n")


db.close()

print("WRDS data pull completed successfully.")
print(f"Files saved in: {raw_dir}")
