import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/Healthcare_Desert_Scored.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("HEALTHCARE DESERT AI — MODEL TRAINING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "Healthcare_Desert_Score"

print("\nTarget variable:")
print(TARGET)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

feature_cols = [

    # Socioeconomic indicators
    "Female_6plus_Ever_Attended_School_pct",
    "Population_Below_15_pct",
    "Sex_Ratio_Females_per_1000_Males",
    "Households_Electricity_pct",
    "Improved_Drinking_Water_pct",
    "Improved_Sanitation_pct",
    "Clean_Cooking_Fuel_pct",
    "Health_Insurance_Coverage_pct",
    "Women_15_49_Literate_pct",

    # Maternal healthcare
    "ANC_4plus_pct",
    "Institutional_Births_pct",
    "Skilled_Birth_Attendance_pct",

    # Child health
    "Under5_Stunted_pct",
    "Children_6_59m_Anaemic_pct",

    # Population
    "Population_2011",
    "Rural_Population_2011",

    # General healthcare infrastructure
    # These are NOT used directly in the Desert Score
    "Sub_Divisional_Hospitals",
    "District_Hospitals"
]
# ============================================================
# 4. PREPARE DATA
# ============================================================

model_df = df[feature_cols + [TARGET]].copy()

print("\nMissing values before preparation:")
print(model_df.isna().sum()[model_df.isna().sum() > 0])


# Replace infinite values
model_df = model_df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# Fill missing numeric values with the median
for col in feature_cols:
    if model_df[col].isna().any():
        model_df[col] = model_df[col].fillna(
            model_df[col].median()
        )

print("\nMissing values after imputation:")
print(model_df[feature_cols].isna().sum().sum())

print("\nDataset after preparation:")
print(model_df.shape)
# ============================================================
# 5. CREATE X AND y
# ============================================================

X = model_df[feature_cols]
y = model_df[TARGET]


print("\nNumber of features:")
print(X.shape[1])

print("\nNumber of observations:")
print(X.shape[0])


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(X_test_scaled)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\nMAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R² Score:", round(linear_r2, 4))


# ============================================================
# 8. RANDOM FOREST REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\nMAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R² Score:", round(rf_r2, 4))


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})

print(
    results.round(4).to_string(index=False)
)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

if rf_r2 > linear_r2:
    best_model = rf_model
    best_model_name = "Random Forest Regression"
else:
    best_model = linear_model
    best_model_name = "Linear Regression"


print("\nBest Model:")
print(best_model_name)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest Regression":

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 60)

    print(
        importance.head(15).round(4).to_string(index=False)
    )

else:

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": linear_model.coef_
    })

    importance["Absolute_Coefficient"] = (
        importance["Coefficient"].abs()
    )

    importance = importance.sort_values(
        "Absolute_Coefficient",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("TOP 15 FEATURES")
    print("=" * 60)

    print(
        importance.head(15).round(4).to_string(index=False)
    )


# ============================================================
# 12. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/healthcare_desert_model.pkl"
)

joblib.dump(
    scaler,
    "models/feature_scaler.pkl"
)

joblib.dump(
    feature_cols,
    "models/feature_columns.pkl"
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

os.makedirs("visualizations/model", exist_ok=True)

results.to_csv(
    "visualizations/model/model_comparison.csv",
    index=False
)

importance.to_csv(
    "visualizations/model/feature_importance.csv",
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nBest model:")
print(best_model_name)

print("\nModel saved to:")
print("models/healthcare_desert_model.pkl")

print("\nResults saved to:")
print("visualizations/model/")

print("\nReady for prediction/deployment.")