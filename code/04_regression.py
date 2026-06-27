from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


# Input and output paths
processed_file = Path("data/processed/panel_with_vars.parquet")
tables_dir = Path("output/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

if not processed_file.exists():
    raise FileNotFoundError(
        "Panel with variables not found. Please run code/03_descriptives.py first."
    )

# Load panel
df = pd.read_parquet(processed_file)

# Variables for this project
DV = "roa"
X_MAIN = "firm_size"
CONTROLS = ["leverage"]

# Keep only complete observations for regression
reg_df = df.dropna(subset=[DV, X_MAIN] + CONTROLS + ["naics"]).copy()

# Industry as categorical control
reg_df["naics"] = reg_df["naics"].astype(str)

# Model 1: simple OLS
model_1 = smf.ols(
    formula=f"{DV} ~ {X_MAIN}",
    data=reg_df
).fit(cov_type="HC3")

# Model 2: with leverage control
model_2 = smf.ols(
    formula=f"{DV} ~ {X_MAIN} + leverage",
    data=reg_df
).fit(cov_type="HC3")

# Model 3: with leverage and industry controls
model_3 = smf.ols(
    formula=f"{DV} ~ {X_MAIN} + leverage + C(naics)",
    data=reg_df
).fit(cov_type="HC3")


def extract_result(model, model_name, variable):
    """Extract coefficient, standard error and p-value from a model."""
    return {
        "model": model_name,
        "variable": variable,
        "coef": model.params.get(variable, None),
        "std_error": model.bse.get(variable, None),
        "p_value": model.pvalues.get(variable, None),
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    }


results = []

for model, name in [
    (model_1, "Model 1: OLS"),
    (model_2, "Model 2: OLS + leverage"),
    (model_3, "Model 3: OLS + leverage + industry"),
]:
    for variable in [X_MAIN, "leverage"]:
        if variable in model.params.index:
            results.append(extract_result(model, name, variable))

results_df = pd.DataFrame(results)
results_df.to_csv(tables_dir / "regression_results.csv", index=False)

# Save full text output as additional file
with open(tables_dir / "regression_results_full.txt", "w") as f:
    f.write("Model 1: OLS\n")
    f.write(model_1.summary().as_text())
    f.write("\n\nModel 2: OLS + leverage\n")
    f.write(model_2.summary().as_text())
    f.write("\n\nModel 3: OLS + leverage + industry\n")
    f.write(model_3.summary().as_text())

print("Regression analysis completed successfully.")
print(f"Sample: {len(reg_df)} firm-year observations")
print(f"Firms: {reg_df['gvkey'].nunique()}")
print(f"Saved: {tables_dir / 'regression_results.csv'}")
print(f"Saved: {tables_dir / 'regression_results_full.txt'}")

beta = model_3.params.get(X_MAIN)
pval = model_3.pvalues.get(X_MAIN)

print("\nH1 diagnostic:")
print(f"Coefficient on firm size: {beta:.4f}")
print(f"p-value: {pval:.4f}")

if beta > 0 and pval < 0.10:
    print("H1 supported: firm size is positively and statistically associated with ROA.")
elif beta > 0:
    print("H1 direction is positive, but not statistically significant.")
else:
    print("H1 not supported: the estimated association is not positive.")

print("\nMain regression table:")
print(results_df)