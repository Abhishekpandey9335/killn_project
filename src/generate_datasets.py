"""
generate_datasets.py
---------------------
AUTOMATION MODULE 1: Data Generation Layer

This script auto-generates 4 linked datasets that mimic real DCS/SCADA
tag exports from a cement kiln, using realistic industry-standard
parameter ranges (based on published cement plant operating norms:
burning zone temp 1400-1480C, kiln speed 2.5-4.5 rpm, free lime 0.5-2.5%,
NOx 400-900 mg/Nm3, etc.)

In a real plant, this script would be replaced by an OPC-UA / SQL
connector pulling live tags. The rest of the automation pipeline
(cleaning -> merging -> feature engineering -> ML -> alerts -> dashboard)
stays IDENTICAL whether the data source is synthetic or live DCS data.
This is what makes the project "automation-ready" for real deployment.

Run: python generate_datasets.py
Output: 4 CSVs written to ../data/
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_HOURS = 24 * 30          # 30 days of data
FREQ_MIN = 5                # one reading every 5 minutes
N = int(N_HOURS * 60 / FREQ_MIN)

timestamps = pd.date_range("2026-06-01", periods=N, freq=f"{FREQ_MIN}min")

os.makedirs("../data", exist_ok=True)

# ---------------------------------------------------------------
# 1. KILN PROCESS DATA
# ---------------------------------------------------------------
kiln_speed = np.clip(np.random.normal(3.5, 0.4, N), 2.5, 4.5)
feed_rate = np.clip(np.random.normal(210, 12, N), 170, 250)          # tph
torque = np.clip(np.random.normal(65, 8, N) + (feed_rate - 210) * 0.15, 40, 95)  # %
burning_zone_temp = np.clip(
    1440 + np.random.normal(0, 15, N) + (feed_rate - 210) * 0.2, 1380, 1500
)
shell_temp = np.clip(np.random.normal(340, 25, N), 280, 420)

kiln_process = pd.DataFrame({
    "timestamp": timestamps,
    "kiln_speed_rpm": kiln_speed.round(2),
    "feed_rate_tph": feed_rate.round(1),
    "kiln_torque_pct": torque.round(1),
    "burning_zone_temp_C": burning_zone_temp.round(1),
    "shell_temp_C": shell_temp.round(1),
})
kiln_process.to_csv("../data/kiln_process.csv", index=False)

# ---------------------------------------------------------------
# 2. COMBUSTION / FUEL DATA
# ---------------------------------------------------------------
coal_feed = np.clip(np.random.normal(14, 1.5, N), 10, 18)             # tph
afr_pct = np.clip(np.random.normal(18, 6, N), 0, 35)                  # % alt fuel substitution
o2_pct = np.clip(np.random.normal(3.2, 0.6, N) - (afr_pct - 18) * 0.01, 1.5, 5.5)
primary_air = np.clip(np.random.normal(10, 1, N), 7, 14)              # % of total air
secondary_air_temp = np.clip(np.random.normal(950, 40, N), 850, 1050)

combustion = pd.DataFrame({
    "timestamp": timestamps,
    "coal_feed_tph": coal_feed.round(2),
    "afr_substitution_pct": afr_pct.round(1),
    "o2_flue_pct": o2_pct.round(2),
    "primary_air_pct": primary_air.round(1),
    "secondary_air_temp_C": secondary_air_temp.round(1),
})
combustion.to_csv("../data/combustion_data.csv", index=False)

# ---------------------------------------------------------------
# 3. EMISSION DATA (depends loosely on combustion + O2)
# ---------------------------------------------------------------
nox = np.clip(600 + (burning_zone_temp - 1440) * 3 - (afr_pct - 18) * 4 + np.random.normal(0, 40, N), 350, 950)
co = np.clip(80 - (o2_pct - 3.2) * 15 + np.random.normal(0, 20, N), 10, 250)
so2 = np.clip(np.random.normal(120, 25, N), 40, 220)
dust = np.clip(np.random.normal(25, 8, N), 5, 60)

emissions = pd.DataFrame({
    "timestamp": timestamps,
    "NOx_mgNm3": nox.round(1),
    "CO_mgNm3": co.round(1),
    "SO2_mgNm3": so2.round(1),
    "dust_mgNm3": dust.round(1),
})
emissions.to_csv("../data/emission_data.csv", index=False)

# ---------------------------------------------------------------
# 4. CLINKER QUALITY DATA (lab data, sampled every ~2 hours)
# ---------------------------------------------------------------
lab_idx = np.arange(0, N, int(120 / FREQ_MIN))          # every 2 hours
lab_timestamps = timestamps[lab_idx]

free_lime = np.clip(
    1.2 - (burning_zone_temp[lab_idx] - 1440) * 0.035
    - (torque[lab_idx] - 65) * 0.01
    + np.random.normal(0, 0.15, len(lab_idx)),
    0.2, 3.5
)
lsf = np.clip(np.random.normal(98, 2, len(lab_idx)), 92, 104)
c3s = np.clip(np.random.normal(58, 4, len(lab_idx)), 45, 68)
silica_ratio = np.clip(np.random.normal(2.5, 0.15, len(lab_idx)), 2.1, 2.9)

quality = pd.DataFrame({
    "timestamp": lab_timestamps,
    "free_lime_pct": free_lime.round(2),
    "LSF": lsf.round(1),
    "C3S_pct": c3s.round(1),
    "silica_ratio": silica_ratio.round(2),
})
quality.to_csv("../data/quality_data.csv", index=False)

print(f"Generated {N} process rows and {len(lab_idx)} lab rows.")
print("Files written to ../data/:")
for f in ["kiln_process.csv", "combustion_data.csv", "emission_data.csv", "quality_data.csv"]:
    print(f" - {f}")
