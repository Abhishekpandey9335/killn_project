# Kiln & Combustion Control Automation System
### AI-Driven Multi-Dataset Automation Pipeline for Cement Kiln Operations

---

## 1. Problem Statement

Cement kilns run on reactive, manual/PID-based control loops — operators
adjust fuel, air, and speed only *after* a deviation is already visible
on the panel. This causes:
- Fuel wastage (over/under-burning)
- Clinker quality swings (free lime out of spec)
- Emission limit breaches (NOx/CO)
- Unplanned refractory wear from undetected kiln instability

**Goal:** Build an automated pipeline that fuses multiple plant datasets,
predicts these issues *before* they happen, and recommends corrective
combustion setpoints in real time.

---

## 2. Why "Automation" is the Core of This Project

This is not a one-off analysis notebook — it is built as a **4-stage
automated pipeline**, where each stage runs independently and feeds the
next automatically:

```
STAGE 1                STAGE 2                 STAGE 3              STAGE 4
generate_datasets.py -> data_pipeline.py   ->  train_models.py  ->  dashboard/app.py
(raw multi-source        (auto-clean, merge,     (auto-train 3        (live monitoring +
 data ingestion)          feature engineer)        ML models)          control recommendation)
```

- **Stage 1** mimics a live DCS/SCADA/OPC-UA data feed (4 independent tag streams).
- **Stage 2** automatically time-aligns and fuses them — no manual merging needed.
- **Stage 3** automatically retrains 3 ML models any time new data arrives.
- **Stage 4** is a live dashboard that reloads models and gives instant
  "what-if" control recommendations.

**In a real plant deployment**, Stage 1 would simply be replaced by a
live OPC-UA/SQL connector pulling actual DCS tags — Stages 2, 3, and 4
require zero changes. That plug-and-play design *is* the automation value.

---

## 3. Multiple Datasets Used (Data Fusion)

| Dataset | Key Parameters | Source Frequency |
|---|---|---|
| Kiln Process | Burning zone temp, kiln speed, torque, feed rate, shell temp | Every 5 min |
| Combustion/Fuel | Coal feed, AFR substitution %, O2%, primary/secondary air | Every 5 min |
| Emissions (CEMS) | NOx, CO, SO2, dust | Every 5 min |
| Clinker Quality (Lab) | Free lime %, LSF, C3S%, silica ratio | Every ~2 hours |

All 4 are fused into a single **master_dataset.csv** with 28 engineered
columns (rolling averages, fuel/air ratios, deviation flags).

*Note: parameter ranges used are based on standard published cement
industry operating norms (burning zone ~1440°C, free lime 0.5–2.5%, NOx
400–900 mg/Nm³ etc.), since live plant-tag data requires DCS access.
The pipeline is built to take real historical CSV exports from JK Cement's
DCS with zero code changes if/when available.*

---

## 4. Models Built

| # | Model | Type | Purpose | Result |
|---|---|---|---|---|
| 1 | Free Lime Predictor | Random Forest Regression | Predict clinker quality before lab result is out | R² = 0.91, MAE = 0.13% |
| 2 | NOx Breach Risk | Gradient Boosting Classifier | Predict emission limit breach risk | Accuracy = 98.5% |
| 3 | Kiln Instability Detector | Isolation Forest (unsupervised) | Flag abnormal torque/temp patterns | Flags ~3% of readings as anomalies |

---

## 5. Dashboard (Live Demo Piece)

`streamlit run dashboard/app.py` opens a browser dashboard with:
- Live KPI cards (current temp, torque, AFR%, NOx, free lime)
- Trend charts across 3 tabs (Process / Combustion / Emissions)
- Anomaly-highlighted burning zone temperature timeline
- **What-If Simulator** — drag sliders for burning zone temp, AFR%,
  coal feed, O2%, kiln speed, torque → instantly see predicted free
  lime % and NOx breach risk, with an automated recommendation
  ("increase AFR substitution", "reduce feed rate", etc.)

This simulator is the "control recommendation engine" — the automation
layer that would eventually write setpoints back to the DCS.

---

## 6. How to Run (Fully Free, Local, PyCharm)

### ⭐ Option A — Single command (runs EVERYTHING end-to-end)
```bash
pip install -r requirements.txt
python run_all.py
```
This one command will: generate all datasets → run the fusion pipeline
→ train all 3 ML models → launch the live dashboard in your browser,
all automatically, one after another.

### Option B — Run each stage manually (for step-by-step demo)
```bash
pip install -r requirements.txt

cd src
python generate_datasets.py
python data_pipeline.py
python train_models.py

cd ../dashboard
streamlit run app.py
```

---

## 7. Tech Stack (100% Free/Open-Source)

Python · Pandas · NumPy · scikit-learn · Streamlit · Plotly · joblib

---

## 8. How to Present This in a Meeting

**Suggested flow (10-12 min):**
1. Problem statement (30 sec) — reactive control costs fuel + quality + compliance risk
2. Show the 4-stage automation architecture diagram (this README, Section 2)
3. Show the 4 datasets and explain data fusion (Section 3)
4. Open the live dashboard → show trend tabs → show anomaly chart
5. **Live demo the What-If simulator** — move AFR slider up, show free
   lime prediction shift and NOx risk drop → this is the "wow" moment
6. Close with: "This same pipeline, pointed at our actual DCS historian
   export, needs zero code changes — Stage 1 is the only swap."

---

## 9. Next Steps for Real Deployment

- Replace `generate_datasets.py` with an OPC-UA/SQL connector to JK
  Cement's DCS (Honeywell/ABB/Siemens) historian
- Add SHAP explainability layer for operator trust
- Add automated email/SMS alerting on anomaly/NOx-risk detection
- Extend to LSTM time-series forecasting for 30-min-ahead burning zone
  temp prediction
