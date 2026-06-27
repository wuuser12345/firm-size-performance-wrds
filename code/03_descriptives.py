from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Input and output paths
processed_file = Path("data/processed/panel_clean.parquet")
tables_dir = Path("output/tables")
figures_dir = Path("output/figures")

tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

if not processed_file.exists():
    raise FileNotFoundError(
        "Clean panel not found. Please run code/02_clean.py first."
    )

# Load clean panel
df = pd.read_parquet(processed_file)

# Keep relevant observations
df = df.dropna(subset=["roa", "firm_size", "leverage"]).copy()

# Optional basic filters
df = df[(df["at"] > 0) & (df["seq"] > 0)].copy()

# Save panel with constructed variables
panel_with_vars = Path("data/processed/panel_with_vars.parquet")
df.to_parquet(panel_with_vars, index=False)

# Summary statistics
vars_for_summary = ["roa", "firm_size", "leverage", "at", "nicon"]

available_vars = [var for var in vars_for_summary if var in df.columns]

summary = (
    df[available_vars]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(4)
)

summary.to_csv(tables_dir / "summary_statistics.csv")

# Correlation matrix
corr_vars = ["roa", "firm_size", "leverage"]
corr = df[corr_vars].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

ax.set_title("Correlation Matrix")
fig.tight_layout()
fig.savefig(figures_dir / "correlation_matrix.png", dpi=300)
plt.close(fig)

# Distribution of dependent variable
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["roa"].dropna(), bins=20)
ax.set_title("Distribution of ROA")
ax.set_xlabel("ROA")
ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(figures_dir / "dv_distribution.png", dpi=300)
plt.close(fig)

# Main relationship: firm size and ROA
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["firm_size"], df["roa"], alpha=0.6)

# Bin means for smoother visual pattern
df_plot = df[["firm_size", "roa"]].dropna().copy()
df_plot["size_bin"] = pd.cut(df_plot["firm_size"], bins=8)
bin_means = df_plot.groupby("size_bin", observed=True)[["firm_size", "roa"]].mean()

ax.plot(bin_means["firm_size"], bin_means["roa"], linewidth=2)
ax.axhline(0, linewidth=0.8, linestyle="--")
ax.set_title("Firm Size and Firm Performance")
ax.set_xlabel("Firm Size: log(Total Assets)")
ax.set_ylabel("Firm Performance: ROA")
fig.tight_layout()
fig.savefig(figures_dir / "main_relationship.png", dpi=300)
plt.close(fig)

# Firm-year counts
firm_years = df.groupby("conm")["fyear"].count().reset_index()
firm_years.columns = ["company", "number_of_firm_year_observations"]
firm_years.to_csv(tables_dir / "firm_year_counts.csv", index=False)

print("Descriptive analysis completed successfully.")
print(f"Sample: {len(df)} firm-year observations")
print(f"Firms: {df['gvkey'].nunique()}")
print(f"Saved: {tables_dir / 'summary_statistics.csv'}")
print(f"Saved: {figures_dir / 'correlation_matrix.png'}")
print(f"Saved: {figures_dir / 'dv_distribution.png'}")
print(f"Saved: {figures_dir / 'main_relationship.png'}")
print(f"Saved: {panel_with_vars}")
print(summary)