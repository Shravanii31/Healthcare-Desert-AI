import pandas as pd
import numpy as np
import os

# ============================================================
# HEALTHCARE DESERT AI — CREATE DESERT SCORE
# ============================================================

INPUT_PATH = "data/Healthcare_Desert_Cleaned.csv"
OUTPUT_PATH = "data/Healthcare_Desert_Scored.csv"

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("HEALTHCARE DESERT AI — DESERT SCORE CREATION")
print("=" * 60)

print(f"\nOriginal dataset shape: {df.shape}")

# ============================================================
# 1. KEEP DISTRICTS WITH RURAL POPULATION
# ============================================================

# Rural healthcare availability cannot be calculated for
# districts with zero rural population.

model_df = df[df["Rural_Population_2011"] > 0].copy()

print(f"\nDistricts with rural population: {len(model_df)}")
print(f"Urban-only districts excluded from score: {len(df) - len(model_df)}")

# ============================================================
# 2. INFRASTRUCTURE VARIABLES
# ============================================================

score_cols = [
    "PHCs_per_100k_rural",
    "CHCs_per_100k_rural",
    "SubCentres_per_100k_rural"
]

print("\nMissing values in score variables:")
print(model_df[score_cols].isna().sum())

# Remove rows where one of the three indicators is unavailable
model_df = model_df.dropna(subset=score_cols).copy()

print(f"\nDistricts available for scoring: {len(model_df)}")

# ============================================================
# 3. CONVERT AVAILABILITY TO PERCENTILE RANK
# ============================================================

# Higher infrastructure availability = better
# Therefore invert percentile rank:
#
# High availability -> low scarcity
# Low availability  -> high scarcity

for col in score_cols:

    percentile = model_df[col].rank(pct=True)

    model_df[col + "_scarcity"] = (1 - percentile) * 100

# ============================================================
# 4. CALCULATE OVERALL DESERT SCORE
# ============================================================

scarcity_cols = [
    "PHCs_per_100k_rural_scarcity",
    "CHCs_per_100k_rural_scarcity",
    "SubCentres_per_100k_rural_scarcity"
]

model_df["Healthcare_Desert_Score"] = (
    model_df[scarcity_cols].mean(axis=1)
)

# ============================================================
# 5. CREATE RISK CATEGORIES
# ============================================================

low_cutoff = model_df["Healthcare_Desert_Score"].quantile(0.33)
high_cutoff = model_df["Healthcare_Desert_Score"].quantile(0.67)

def classify_risk(score):

    if score >= high_cutoff:
        return "High Risk"

    elif score >= low_cutoff:
        return "Medium Risk"

    else:
        return "Low Risk"

model_df["Healthcare_Desert_Risk"] = (
    model_df["Healthcare_Desert_Score"]
    .apply(classify_risk)
)

# ============================================================
# 6. SAVE RESULTS
# ============================================================

os.makedirs("data", exist_ok=True)

model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DESERT SCORE SUMMARY")
print("=" * 60)

print(
    model_df["Healthcare_Desert_Score"]
    .describe()
    .round(2)
)

print("\nRisk Category Distribution:")
print(
    model_df["Healthcare_Desert_Risk"]
    .value_counts()
)

print("\nRisk Category Percentage:")
print(
    (
        model_df["Healthcare_Desert_Risk"]
        .value_counts(normalize=True) * 100
    ).round(2)
)

print("\nRisk Cutoffs:")
print(f"Low/Medium cutoff: {low_cutoff:.2f}")
print(f"Medium/High cutoff: {high_cutoff:.2f}")

print("\nTop 15 Highest-Risk Districts:")

display_cols = [
    "District",
    "State_UT",
    "Rural_Population_2011",
    "PHCs_per_100k_rural",
    "CHCs_per_100k_rural",
    "SubCentres_per_100k_rural",
    "Healthcare_Desert_Score",
    "Healthcare_Desert_Risk"
]

print(
    model_df
    .sort_values("Healthcare_Desert_Score", ascending=False)
    [display_cols]
    .head(15)
    .round(2)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("DESERT SCORE CREATION COMPLETE")
print("=" * 60)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print(f"\nFinal scored dataset shape: {model_df.shape}")