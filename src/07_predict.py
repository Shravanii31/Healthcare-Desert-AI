import pandas as pd
import numpy as np
import joblib


# ============================================================
# HEALTHCARE DESERT AI — PREDICTION
# ============================================================

MODEL_PATH = "models/healthcare_desert_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


print("=" * 60)
print("HEALTHCARE DESERT AI — DISTRICT PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print("\nModel loaded successfully.")
print("Number of features:", len(feature_columns))


# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_PATH = "data/Healthcare_Desert_Scored.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 3. SELECT A DISTRICT
# ============================================================

district_name = input("\nEnter district name: ").strip()

matches = df[
    df["District"].str.lower() == district_name.lower()
]

if len(matches) == 0:
    print("\nDistrict not found.")
    exit()

row = matches.iloc[0]


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

X = row[feature_columns].to_frame().T

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(df[feature_columns].median())

X_scaled = scaler.transform(X)


# ============================================================
# 5. PREDICT
# ============================================================

prediction = model.predict(X_scaled)[0]

prediction = max(0, min(100, prediction))


# ============================================================
# 6. RISK CATEGORY
# ============================================================

if prediction < 36.93:
    risk = "Low Risk"
elif prediction < 63.52:
    risk = "Medium Risk"
else:
    risk = "High Risk"


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nDistrict:", row["District"])
print("State/UT:", row["State_UT"])

print("\nPredicted Healthcare Desert Score:")
print(round(prediction, 2))

print("\nHealthcare Desert Risk:")
print(risk)

print("\n" + "=" * 60)