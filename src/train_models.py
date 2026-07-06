"""
train_models.py
-----------------
AUTOMATION MODULE 3: ML Model Training Layer

Trains 3 models automatically from the master dataset:
  1. RandomForestRegressor  -> predicts free_lime_pct (clinker quality)
  2. GradientBoostingClassifier -> predicts NOx breach risk (emissions)
  3. IsolationForest -> flags kiln instability / anomalies (unsupervised)

All trained models are saved to ../outputs/ as .pkl files so the
dashboard can load them instantly without retraining - this is what
makes the system "automation-ready": generate data -> pipeline -> train
once -> dashboard just loads + predicts continuously.

Run: python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

DATA_PATH = "../data/master_dataset.csv"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "kiln_speed_rpm", "feed_rate_tph", "kiln_torque_pct",
    "burning_zone_temp_C", "shell_temp_C",
    "coal_feed_tph", "afr_substitution_pct", "o2_flue_pct",
    "primary_air_pct", "secondary_air_temp_C",
    "burning_zone_temp_C_roll1h", "kiln_torque_pct_roll1h",
    "coal_afr_ratio", "air_fuel_ratio", "bz_temp_deviation", "torque_volatility",
]


def train_free_lime_model(df):
    print("\n--- Model 1: Free Lime % Prediction (Regression) ---")
    X = df[FEATURES]
    y = df["free_lime_pct"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.3f} | R2 Score: {r2:.3f}")

    joblib.dump(model, f"{OUT_DIR}/free_lime_model.pkl")

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Top 5 influential features:\n", importance.head(5))
    return {"mae": mae, "r2": r2}


def train_nox_risk_model(df):
    print("\n--- Model 2: NOx Breach Risk Prediction (Classification) ---")
    X = df[FEATURES]
    y = df["nox_breach_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, f"{OUT_DIR}/nox_risk_model.pkl")
    return {"accuracy": acc}


def train_anomaly_model(df):
    print("\n--- Model 3: Kiln Instability Anomaly Detection (Unsupervised) ---")
    X = df[["kiln_torque_pct", "burning_zone_temp_C", "shell_temp_C", "torque_volatility"]]

    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X)

    scores = model.predict(X)   # -1 = anomaly, 1 = normal
    n_anomalies = (scores == -1).sum()
    print(f"Detected {n_anomalies} anomalous readings out of {len(X)} ({n_anomalies/len(X)*100:.2f}%)")

    joblib.dump(model, f"{OUT_DIR}/anomaly_model.pkl")
    return {"n_anomalies": int(n_anomalies)}


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURES + ["free_lime_pct", "nox_breach_risk"])

    # Free lime is a LAB measurement taken every ~2 hours. Training on every
    # 5-min row repeats the same lab value against many rows whose process
    # conditions already drifted since the sample was taken, so we train
    # only on rows at the actual sampling instant (change-points in free_lime_pct).
    lab_rows = df[df["free_lime_pct"].ne(df["free_lime_pct"].shift())].copy()
    print(f"\nUsing {len(lab_rows)} actual lab-sample rows (out of {len(df)} total) for free-lime model.")

    results = {}
    results["free_lime"] = train_free_lime_model(lab_rows)
    results["nox_risk"] = train_nox_risk_model(df)
    results["anomaly"] = train_anomaly_model(df)

    print("\n=== ALL MODELS TRAINED & SAVED TO ../outputs/ ===")
    print(results)
