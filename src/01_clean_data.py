import pandas as pd
import numpy as np
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/Master_Healthcare_Desert_Dataset_Working.csv"
OUTPUT_PATH = "data/Healthcare_Desert_Cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("HEALTHCARE DESERT AI — DATA CLEANING")
print("=" * 60)

print(f"\nOriginal shape: {df.shape}")


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# Remove extra spaces from text columns
for col in ["District", "State_UT"]:
    df[col] = df[col].astype(str).str.strip()


# ============================================================
# 3. HANDLE INVALID PERCENTAGE VALUES
# ============================================================

percentage_cols = [
    "Female_6plus_Ever_Attended_School_pct",
    "Population_Below_15_pct",
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
    "Diarrhoea_Taken_to_Health_Facility_pct",
    "ARI_Taken_to_Health_Facility_pct",
    "Under5_Stunted_pct",
    "Children_6_59m_Anaemic_pct"
]

print("\n" + "=" * 60)
print("INVALID PERCENTAGE VALUES")
print("=" * 60)

for col in percentage_cols:
    invalid = ((df[col] < 0) | (df[col] > 100)).sum()
    if invalid > 0:
        print(f"{col}: {invalid} invalid values")

        # Convert invalid values to missing
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan


# ============================================================
# 4. HANDLE INFINITE VALUES
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns

print("\n" + "=" * 60)
print("INFINITE VALUES")
print("=" * 60)

inf_counts = np.isinf(df[numeric_cols]).sum()
inf_counts = inf_counts[inf_counts > 0]

print(inf_counts)

# Replace infinity with NaN
df[numeric_cols] = df[numeric_cols].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 5. MISSING VALUES SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

print("\nMissing Value Percentage:")

missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

print(missing_pct.round(2))


# ============================================================
# 6. SAVE CLEAN DATASET
# ============================================================

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print("\nClean dataset saved to:")
print(OUTPUT_PATH)

print(f"\nFinal shape: {df.shape}")