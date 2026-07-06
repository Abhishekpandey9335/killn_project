"""
data_pipeline.py
------------------
AUTOMATION MODULE 2: Data Fusion & Feature Engineering Pipeline

Takes the 4 raw datasets (process, combustion, emissions, quality)
and automatically:
  1. Loads and time-aligns them
  2. Forward-fills lab data (since lab samples are sparse - every 2 hrs)
  3. Merges everything into a single master dataframe on timestamp
  4. Engineers rolling-average and ratio features used by the ML models
  5. Saves the final master dataset for reuse by any model script

This script is the "single source of truth" - every model/dashboard
script downstream reads master_dataset.csv, so the pipeline only
needs to run once per data refresh cycle. In production this would
be scheduled (cron / Airflow) to run every N minutes automatically.

Run: python data_pipeline.py
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "../data"

def load_and_merge():
    kiln = pd.read_csv(f"{DATA_DIR}/kiln_process.csv", parse_dates=["timestamp"])
    comb = pd.read_csv(f"{DATA_DIR}/combustion_data.csv", parse_dates=["timestamp"])
    emis = pd.read_csv(f"{DATA_DIR}/emission_data.csv", parse_dates=["timestamp"])
    qual = pd.read_csv(f"{DATA_DIR}/quality_data.csv", parse_dates=["timestamp"])

    df = kiln.merge(comb, on="timestamp", how="inner")
    df = df.merge(emis, on="timestamp", how="inner")

    # lab data is sparse -> merge_asof to bring forward last known lab reading
    df = df.sort_values("timestamp")
    qual = qual.sort_values("timestamp")
    df = pd.merge_asof(df, qual, on="timestamp", direction="backward")

    return df


def engineer_features(df):
    df = df.copy()

    # Rolling averages (last 1 hour = 12 readings @ 5 min)
    roll_cols = ["burning_zone_temp_C", "kiln_torque_pct", "o2_flue_pct", "NOx_mgNm3"]
    for col in roll_cols:
        df[f"{col}_roll1h"] = df[col].rolling(window=12, min_periods=1).mean()

    # Fuel / air ratios - key combustion efficiency indicators
    df["coal_afr_ratio"] = df["coal_feed_tph"] / (df["afr_substitution_pct"] + 1)
    df["air_fuel_ratio"] = df["primary_air_pct"] / (df["coal_feed_tph"] + 0.1)

    # Deviation from target burning zone temp (1440C is typical setpoint)
    df["bz_temp_deviation"] = df["burning_zone_temp_C"] - 1440

    # Instability flag: torque or temp fluctuating fast -> feature for anomaly model
    df["torque_volatility"] = df["kiln_torque_pct"].rolling(window=6, min_periods=1).std().fillna(0)

    # Emission risk flag (rule-based label, useful for classification demo)
    df["nox_breach_risk"] = (df["NOx_mgNm3"] > 750).astype(int)

    return df


def run_pipeline():
    print("Loading raw datasets...")
    df = load_and_merge()
    print(f"Merged shape: {df.shape}")

    print("Engineering features...")
    df = engineer_features(df)

    df.to_csv(f"{DATA_DIR}/master_dataset.csv", index=False)
    print(f"Master dataset saved -> {DATA_DIR}/master_dataset.csv")
    print(f"Final shape: {df.shape}, Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    run_pipeline()
