import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# HEALTHCARE DESERT AI — DESERT SCORE EDA
# ============================================================

INPUT_PATH = "data/Healthcare_Desert_Scored.csv"
OUTPUT_DIR = "visualizations/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("HEALTHCARE DESERT AI — DESERT SCORE EDA")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

# ============================================================
# 1. RISK DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("RISK CATEGORY DISTRIBUTION")
print("=" * 60)

print(df["Healthcare_Desert_Risk"].value_counts())
print("\nPercentage:")
print(
    (df["Healthcare_Desert_Risk"].value_counts(normalize=True) * 100)
    .round(2)
)

# ============================================================
# 2. DESERT SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("DESERT SCORE STATISTICS")
print("=" * 60)

print(
    df["Healthcare_Desert_Score"]
    .describe()
    .round(2)
)

plt.figure(figsize=(10, 6))

plt.hist(
    df["Healthcare_Desert_Score"],
    bins=30
)

plt.axvline(
    df["Healthcare_Desert_Score"].quantile(0.33),
    linestyle="--",
    label="Low/Medium cutoff"
)

plt.axvline(
    df["Healthcare_Desert_Score"].quantile(0.67),
    linestyle="--",
    label="Medium/High cutoff"
)

plt.title("Distribution of Healthcare Desert Score")
plt.xlabel("Healthcare Desert Score")
plt.ylabel("Number of Districts")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/desert_score_distribution.png",
    dpi=150
)

plt.close()

# ============================================================
# 3. STATE-WISE RISK
# ============================================================

state_risk = pd.crosstab(
    df["State_UT"],
    df["Healthcare_Desert_Risk"]
)

for col in ["Low Risk", "Medium Risk", "High Risk"]:
    if col not in state_risk.columns:
        state_risk[col] = 0

state_risk["Total"] = state_risk.sum(axis=1)

state_risk["High_Risk_Pct"] = (
    state_risk["High Risk"] /
    state_risk["Total"] * 100
)

state_risk = state_risk.sort_values(
    "High_Risk_Pct",
    ascending=False
)

print("\n" + "=" * 60)
print("STATES WITH HIGHEST SHARE OF HIGH-RISK DISTRICTS")
print("=" * 60)

print(
    state_risk[
        ["Low Risk", "Medium Risk", "High Risk", "Total", "High_Risk_Pct"]
    ]
    .round(2)
    .head(15)
)

state_risk.to_csv(
    f"{OUTPUT_DIR}/state_risk_summary.csv"
)

# ============================================================
# 4. AVERAGE DESERT SCORE BY STATE
# ============================================================

state_score = (
    df.groupby("State_UT")
    ["Healthcare_Desert_Score"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("\n" + "=" * 60)
print("STATES BY AVERAGE HEALTHCARE DESERT SCORE")
print("=" * 60)

print(state_score.round(2).head(15))

state_score.to_csv(
    f"{OUTPUT_DIR}/state_desert_score_summary.csv"
)

# ============================================================
# 5. TOP 20 HIGHEST-RISK DISTRICTS
# ============================================================

display_cols = [
    "District",
    "State_UT",
    "Healthcare_Desert_Score",
    "Healthcare_Desert_Risk",
    "PHCs_per_100k_rural",
    "CHCs_per_100k_rural",
    "SubCentres_per_100k_rural"
]

print("\n" + "=" * 60)
print("TOP 20 HIGHEST-RISK DISTRICTS")
print("=" * 60)

top20 = (
    df.sort_values(
        "Healthcare_Desert_Score",
        ascending=False
    )
    [display_cols]
    .head(20)
)

print(top20.round(2).to_string(index=False))

top20.to_csv(
    f"{OUTPUT_DIR}/top20_highest_risk_districts.csv",
    index=False
)

# ============================================================
# 6. TOP 20 LOWEST-RISK DISTRICTS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 LOWEST-RISK DISTRICTS")
print("=" * 60)

bottom20 = (
    df.sort_values(
        "Healthcare_Desert_Score",
        ascending=True
    )
    [display_cols]
    .head(20)
)

print(bottom20.round(2).to_string(index=False))

bottom20.to_csv(
    f"{OUTPUT_DIR}/top20_lowest_risk_districts.csv",
    index=False
)

# ============================================================
# 7. SOCIOECONOMIC CORRELATIONS
# ============================================================

analysis_cols = [
    "Healthcare_Desert_Score",
    "Population_Below_15_pct",
    "Sex_Ratio_Females_per_1000_Males",
    "Households_Electricity_pct",
    "Improved_Drinking_Water_pct",
    "Improved_Sanitation_pct",
    "Clean_Cooking_Fuel_pct",
    "Health_Insurance_Coverage_pct",
    "Women_15_49_Literate_pct",
    "ANC_4plus_pct",
    "Institutional_Births_pct",
    "Skilled_Birth_Attendance_pct",
    "Fully_Vaccinated_12_23m_pct",
    "Under5_Stunted_pct",
    "Children_6_59m_Anaemic_pct"
]

correlation = (
    df[analysis_cols]
    .corr()["Healthcare_Desert_Score"]
    .sort_values()
)

print("\n" + "=" * 60)
print("CORRELATION WITH HEALTHCARE DESERT SCORE")
print("=" * 60)

print(correlation.round(2))

correlation.to_csv(
    f"{OUTPUT_DIR}/desert_score_correlations.csv"
)

# ============================================================
# 8. SCORE VS HEALTH INSURANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Health_Insurance_Coverage_pct"],
    df["Healthcare_Desert_Score"]
)

plt.title(
    "Healthcare Insurance Coverage vs Healthcare Desert Score"
)

plt.xlabel("Health Insurance Coverage (%)")
plt.ylabel("Healthcare Desert Score")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/insurance_vs_desert_score.png",
    dpi=150
)

plt.close()

# ============================================================
# 9. SCORE VS LITERACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Women_15_49_Literate_pct"],
    df["Healthcare_Desert_Score"]
)

plt.title(
    "Women's Literacy vs Healthcare Desert Score"
)

plt.xlabel("Women 15–49 Literate (%)")
plt.ylabel("Healthcare Desert Score")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/literacy_vs_desert_score.png",
    dpi=150
)

plt.close()

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DESERT SCORE EDA COMPLETE!")
print("=" * 60)

print("\nOutputs saved to:")
print(OUTPUT_DIR)