import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA AND MODEL
# ============================================================

DATA_PATH = "data/Healthcare_Desert_Scored.csv"
MODEL_PATH = "models/healthcare_desert_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_PATH)

TARGET = "Healthcare_Desert_Score"

print("=" * 60)
print("HEALTHCARE DESERT AI — MODEL EVALUATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nModel loaded:")
print(MODEL_PATH)

print("\nNumber of features:")
print(len(feature_cols))


# ============================================================
# 2. PREPARE DATA
# ============================================================

model_df = df[feature_cols + [TARGET]].copy()

# Replace infinite values
model_df = model_df.replace([np.inf, -np.inf], np.nan)

# Median imputation — same approach as training
for col in feature_cols:
    if model_df[col].isna().any():
        model_df[col] = model_df[col].fillna(
            model_df[col].median()
        )

X = model_df[feature_cols]
y = model_df[TARGET]


# ============================================================
# 3. SAME TRAIN-TEST SPLIT AS TRAINING
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
# 4. PREDICT TEST SET
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nMAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))


# ============================================================
# 6. PREDICTION RESULTS
# ============================================================

results = df.loc[X_test.index, [
    "District",
    "State_UT",
    TARGET,
    "Healthcare_Desert_Risk"
]].copy()

results["Predicted_Desert_Score"] = predictions

results["Prediction_Error"] = (
    results["Predicted_Desert_Score"]
    - results[TARGET]
)

results["Absolute_Error"] = (
    results["Prediction_Error"]
    .abs()
)

results = results.sort_values(
    "Healthcare_Desert_Score",
    ascending=False
)


# ============================================================
# 7. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "visualizations/model",
    exist_ok=True
)


# ============================================================
# 8. SAVE PREDICTIONS
# ============================================================

results.to_csv(
    "visualizations/model/model_predictions.csv",
    index=False
)

print("\nPrediction results saved to:")
print("visualizations/model/model_predictions.csv")


# ============================================================
# 9. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    predictions
)

# Perfect prediction line
min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.title(
    "Actual vs Predicted Healthcare Desert Score"
)

plt.xlabel(
    "Actual Healthcare Desert Score"
)

plt.ylabel(
    "Predicted Healthcare Desert Score"
)

plt.tight_layout()

plt.savefig(
    "visualizations/model/actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 10. PREDICTION ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    results["Prediction_Error"],
    bins=20
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "Distribution of Model Prediction Errors"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Number of Districts"
)

plt.tight_layout()

plt.savefig(
    "visualizations/model/prediction_error_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. TOP PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results[
        [
            "District",
            "State_UT",
            TARGET,
            "Predicted_Desert_Score",
            "Healthcare_Desert_Risk",
            "Absolute_Error"
        ]
    ]
    .head(15)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 12. ERROR SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ERROR SUMMARY")
print("=" * 60)

print(
    results["Absolute_Error"]
    .describe()
    .round(2)
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)

print("\nFiles saved inside:")
print("visualizations/model/")