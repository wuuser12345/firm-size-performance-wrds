from pathlib import Path
from datetime import datetime
import os

import pandas as pd
import wrds
from dotenv import load_dotenv


# Load WRDS login from local .env file
load_dotenv()

WRDS_USERNAME = os.getenv("WRDS_USERNAME")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD")

if WRDS_USERNAME is None:
    raise ValueError("WRDS_USERNAME is missing. Please check your .env file.")

if WRDS_PASSWORD is None:
    raise ValueError("WRDS_PASSWORD is missing. Please check your .env file.")


# ------------------------------------------------------------
# European Manufacturing SME sample
# SME filter: AT <= 43 million
# Manufacturing sector: NAICS codes starting with 31, 32 or 33
# ------------------------------------------------------------

EUROPEAN_COUNTRIES = [
    "AUT",  # Austria
    "BEL",  # Belgium
    "CHE",  # Switzerland
    "DEU",  # Germany
    "DNK",  # Denmark
    "ESP",  # Spain
    "FIN",  # Finland
    "FRA",  # France
    "GBR",  # United Kingdom
    "IRL",  # Ireland
    "ITA",  # Italy
    "NLD",  # Netherlands
    "NOR",  # Norway
    "POL",  # Poland
    "PRT",  # Portugal
    "SWE",  # Sweden
]

country_string = ", ".join([f"'{country}'" for country in EUROPEAN_COUNTRIES])


# Variables for this research note
VARIABLES = [
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
    "naicsh as naics",
]

variable_string = ", ".join(VARIABLES)


# Create output folders
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
raw_dir = Path("data/raw")
timestamped_dir = raw_dir / timestamp

raw_dir.mkdir(parents=True, exist_ok=True)
timestamped_dir.mkdir(parents=True, exist_ok=True)


print("Connecting to WRDS...")

db = wrds.Connection(
    wrds_username=WRDS_USERNAME,
    wrds_password=WRDS_PASSWORD,
)

print("Connected to WRDS.")
print("Pulling European manufacturing SME data from Compustat Global...")


query = f"""
    SELECT {variable_string}
    FROM comp_global_daily.g_funda
    WHERE fyear BETWEEN 2015 AND 2024
    AND fic IN ({country_string})
    AND at > 0.1
    AND at <= 43
    AND seq > 0
    AND (
        CAST(naicsh AS TEXT) LIKE '31%%'
        OR CAST(naicsh AS TEXT) LIKE '32%%'
        OR CAST(naicsh AS TEXT) LIKE '33%%'
    )
"""

df = db.raw_sql(query)

db.close()


# Save timestamped backup
timestamped_file = timestamped_dir / "wrds_pull.parquet"
df.to_parquet(timestamped_file, index=False)

# Save main file used by 02_clean.py
main_file = raw_dir / "wrds_manual_pull.csv"
df.to_csv(main_file, index=False)


# Save column schema
schema = pd.DataFrame(
    {
        "column": [
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
            "naics",
        ],
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
            "North American Industry Classification Code",
        ],
    }
)

schema.to_csv(timestamped_dir / "column_schema.csv", index=False)


# Save metadata
with open(timestamped_dir / "pull_metadata.txt", "w") as f:
    f.write("WRDS / Compustat Global Daily - Fundamentals Annual Pull\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write("Table: comp_global_daily.g_funda\n")
    f.write("Fiscal years: 2015-2024\n")
    f.write("Sample: European Manufacturing SMEs\n")
    f.write("SME filter: at > 0.1 and at <= 43, seq > 0\n")
    f.write("Manufacturing filter: NAICS starts with 31, 32 or 33\n\n")

    f.write("Countries:\n")
    for country in EUROPEAN_COUNTRIES:
        f.write(f"- {country}\n")

    f.write("\nVariables:\n")
    for var in VARIABLES:
        f.write(f"- {var}\n")

    f.write("\nSummary:\n")
    f.write(f"Rows downloaded: {len(df)}\n")
    f.write(f"Firms downloaded: {df['gvkey'].nunique()}\n")


print("WRDS data pull completed successfully.")
print(f"Rows downloaded: {len(df)}")
print(f"Firms downloaded: {df['gvkey'].nunique()}")
print(f"Timestamped backup saved to: {timestamped_file}")
print(f"Main CSV saved to: {main_file}")

print("\nCountries in sample:")
print(df["fic"].value_counts().sort_index())

print("\nFirst companies downloaded:")
print(df[["gvkey", "conm", "fic", "naics"]].drop_duplicates().head(20))