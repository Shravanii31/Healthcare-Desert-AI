import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/Healthcare_Desert_Cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("HEALTHCARE DESERT AI — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of States/UTs:")
print(df["State_UT"].nunique())

print("\nNumber of Districts:")
print(df["District"].nunique())


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

print("\nMissing Value Percentage:")

missing_pct = (df.isnull().mean() * 100)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

print(missing_pct.round(2))


# ============================================================
# 3. CHECK INFINITE VALUES
# ============================================================

print("\n" + "=" * 60)
print("INFINITE VALUES")
print("=" * 60)

numeric_df = df.select_dtypes(include=np.number)

infinite_counts = np.isinf(numeric_df).sum()
infinite_counts = infinite_counts[infinite_counts > 0]

print(infinite_counts)


# ============================================================
# 4. HEALTHCARE INFRASTRUCTURE SUMMARY
# ============================================================

healthcare_cols = [
    "Sub_Centres",
    "PHCs",
    "CHCs",
    "Sub_Divisional_Hospitals",
    "District_Hospitals",
    "PHCs_per_100k_rural",
    "CHCs_per_100k_rural",
    "SubCentres_per_100k_rural",
    "PHC_coverage_vs_general_norm",
    "CHC_coverage_vs_general_norm"
]

print("\n" + "=" * 60)
print("HEALTHCARE INFRASTRUCTURE SUMMARY")
print("=" * 60)

print(df[healthcare_cols].describe().round(2))


# ============================================================
# 5. TOP STATES BY PHC AVAILABILITY
# ============================================================

state_summary = df.groupby("State_UT").agg(
    Districts=("District", "count"),
    Total_PHCs=("PHCs", "sum"),
    Total_CHCs=("CHCs", "sum"),
    Total_Sub_Centres=("Sub_Centres", "sum"),
    Avg_PHC_per_100k=("PHCs_per_100k_rural", "mean"),
    Avg_CHC_per_100k=("CHCs_per_100k_rural", "mean")
).reset_index()

print("\n" + "=" * 60)
print("STATES WITH HIGHEST AVERAGE PHC AVAILABILITY")
print("=" * 60)

print(
    state_summary
    .sort_values("Avg_PHC_per_100k", ascending=False)
    .head(10)
    .round(2)
)


# ============================================================
# 6. CREATE EDA OUTPUT FOLDER
# ============================================================

os.makedirs("visualizations/eda", exist_ok=True)


# ============================================================
# 7. DISTRIBUTION — PHCs PER 100K RURAL POPULATION
# ============================================================

plt.figure(figsize=(10, 6))

phc_data = df["PHCs_per_100k_rural"].replace([np.inf, -np.inf], np.nan).dropna()

plt.hist(phc_data, bins=30)

plt.title("Distribution of PHCs per 100,000 Rural Population")
plt.xlabel("PHCs per 100,000 Rural Population")
plt.ylabel("Number of Districts")

plt.tight_layout()

plt.savefig(
    "visualizations/eda/phc_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# 8. DISTRIBUTION — CHCs PER 100K
# ============================================================

plt.figure(figsize=(10, 6))

chc_data = df["CHCs_per_100k_rural"].replace([np.inf, -np.inf], np.nan).dropna()

plt.hist(chc_data, bins=30)

plt.title("Distribution of CHCs per 100,000 Rural Population")
plt.xlabel("CHCs per 100,000 Rural Population")
plt.ylabel("Number of Districts")

plt.tight_layout()

plt.savefig(
    "visualizations/eda/chc_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# 9. RURAL POPULATION VS PHCs
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Rural_Population_2011"],
    df["PHCs"]
)

plt.title("Rural Population vs Number of PHCs")
plt.xlabel("Rural Population (2011)")
plt.ylabel("Number of PHCs")

plt.tight_layout()

plt.savefig(
    "visualizations/eda/rural_population_vs_phcs.png",
    dpi=150
)

plt.show()


# ============================================================
# 10. HEALTHCARE INFRASTRUCTURE CORRELATION
# ============================================================

corr_cols = [
    "Rural_Population_2011",
    "Sub_Centres",
    "PHCs",
    "CHCs",
    "Sub_Divisional_Hospitals",
    "District_Hospitals",
    "PHCs_per_100k_rural",
    "CHCs_per_100k_rural",
    "SubCentres_per_100k_rural",
    "PHC_coverage_vs_general_norm",
    "CHC_coverage_vs_general_norm"
]

correlation = df[corr_cols].replace(
    [np.inf, -np.inf],
    np.nan
).corr()

print("\n" + "=" * 60)
print("HEALTHCARE INFRASTRUCTURE CORRELATION")
print("=" * 60)

print(correlation.round(2))


# ============================================================
# 11. SAVE STATE SUMMARY
# ============================================================

state_summary.to_csv(
    "visualizations/eda/state_healthcare_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)

print("\nEDA files saved inside:")
print("visualizations/eda/")