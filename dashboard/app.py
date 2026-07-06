"""
app.py
-------
AUTOMATION MODULE 4: Real-Time Monitoring Dashboard (Streamlit)

This is the presentation layer. It loads the master dataset + the
3 trained models and shows:
  - Live-style trend charts (kiln temp, torque, emissions)
  - Free lime prediction vs actual
  - NOx breach risk gauge
  - Anomaly flags on a timeline
  - A simple "what-if" simulator: change combustion inputs and see
    predicted free lime + NOx risk instantly (this is the automation
    "control recommendation" piece)

Run from the dashboard/ folder:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Kiln & Combustion Automation Dashboard", layout="wide")

DATA_PATH = "../data/master_dataset.csv"
MODEL_DIR = "../outputs"

FEATURES = [
    "kiln_speed_rpm", "feed_rate_tph", "kiln_torque_pct",
    "burning_zone_temp_C", "shell_temp_C",
    "coal_feed_tph", "afr_substitution_pct", "o2_flue_pct",
    "primary_air_pct", "secondary_air_temp_C",
    "burning_zone_temp_C_roll1h", "kiln_torque_pct_roll1h",
    "coal_afr_ratio", "air_fuel_ratio", "bz_temp_deviation", "torque_volatility",
]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df

@st.cache_resource
def load_models():
    free_lime_model = joblib.load(f"{MODEL_DIR}/free_lime_model.pkl")
    nox_model = joblib.load(f"{MODEL_DIR}/nox_risk_model.pkl")
    anomaly_model = joblib.load(f"{MODEL_DIR}/anomaly_model.pkl")
    return free_lime_model, nox_model, anomaly_model

df = load_data()
free_lime_model, nox_model, anomaly_model = load_models()

st.title("🏭 Kiln & Combustion Control — Automation Dashboard")
st.caption("Multi-dataset automation pipeline: Process + Combustion + Emissions + Quality data fused into a single AI-driven monitoring & control-recommendation system.")

# ---------------- Top KPI row ----------------
latest = df.iloc[-1]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Burning Zone Temp", f"{latest['burning_zone_temp_C']:.1f} °C")
col2.metric("Kiln Torque", f"{latest['kiln_torque_pct']:.1f} %")
col3.metric("AFR Substitution", f"{latest['afr_substitution_pct']:.1f} %")
col4.metric("NOx", f"{latest['NOx_mgNm3']:.0f} mg/Nm³")
col5.metric("Free Lime (last lab)", f"{latest['free_lime_pct']:.2f} %")

st.divider()

# ---------------- Trend charts ----------------
st.subheader("📈 Process & Emission Trends (last 3 days)")
recent = df.tail(864)  # last 3 days @5min

tab1, tab2, tab3 = st.tabs(["Kiln Process", "Combustion", "Emissions"])

with tab1:
    fig = px.line(recent, x="timestamp", y=["burning_zone_temp_C", "shell_temp_C"],
                  title="Kiln Temperatures")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(recent, x="timestamp", y="kiln_torque_pct", title="Kiln Torque %")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = px.line(recent, x="timestamp", y=["coal_feed_tph", "afr_substitution_pct"],
                   title="Fuel Feed: Coal vs AFR Substitution")
    st.plotly_chart(fig3, use_container_width=True)
    fig4 = px.line(recent, x="timestamp", y="o2_flue_pct", title="Flue Gas O2 %")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    fig5 = px.line(recent, x="timestamp", y=["NOx_mgNm3", "CO_mgNm3", "SO2_mgNm3"],
                   title="Emissions")
    fig5.add_hline(y=750, line_dash="dash", line_color="red", annotation_text="NOx Limit Risk Threshold")
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ---------------- Anomaly detection ----------------
st.subheader("🚨 Kiln Instability Anomaly Detection")
anomaly_features = df[["kiln_torque_pct", "burning_zone_temp_C", "shell_temp_C", "torque_volatility"]]
df["anomaly_flag"] = anomaly_model.predict(anomaly_features)
recent_anom = df.tail(864)

fig6 = px.scatter(recent_anom, x="timestamp", y="burning_zone_temp_C",
                   color=recent_anom["anomaly_flag"].map({1: "Normal", -1: "Anomaly"}),
                   color_discrete_map={"Normal": "steelblue", "Anomaly": "red"},
                   title="Burning Zone Temp — Anomalies Highlighted")
st.plotly_chart(fig6, use_container_width=True)

n_anom_recent = (recent_anom["anomaly_flag"] == -1).sum()
st.info(f"⚠️ {n_anom_recent} anomalous readings detected in the last 3 days — recommend refractory/torque inspection if cluster persists.")

st.divider()

# ---------------- What-if simulator (control recommendation) ----------------
st.subheader("🎛️ What-If Combustion Simulator (Control Recommendation Engine)")
st.caption("Adjust combustion parameters below to see predicted free lime % and NOx breach risk — this is the core 'automation' piece that would drive setpoint recommendations in a live system.")

c1, c2, c3 = st.columns(3)
with c1:
    bz_temp = st.slider("Burning Zone Temp (°C)", 1380, 1500, int(latest["burning_zone_temp_C"]))
    kiln_speed = st.slider("Kiln Speed (rpm)", 2.5, 4.5, float(latest["kiln_speed_rpm"]))
with c2:
    afr = st.slider("AFR Substitution (%)", 0, 35, int(latest["afr_substitution_pct"]))
    coal = st.slider("Coal Feed (tph)", 10.0, 18.0, float(latest["coal_feed_tph"]))
with c3:
    o2 = st.slider("Flue O2 (%)", 1.5, 5.5, float(latest["o2_flue_pct"]))
    torque = st.slider("Kiln Torque (%)", 40, 95, int(latest["kiln_torque_pct"]))

sim_row = latest.copy()
sim_row["burning_zone_temp_C"] = bz_temp
sim_row["kiln_speed_rpm"] = kiln_speed
sim_row["afr_substitution_pct"] = afr
sim_row["coal_feed_tph"] = coal
sim_row["o2_flue_pct"] = o2
sim_row["kiln_torque_pct"] = torque
sim_row["bz_temp_deviation"] = bz_temp - 1440
sim_row["coal_afr_ratio"] = coal / (afr + 1)
sim_row["air_fuel_ratio"] = sim_row["primary_air_pct"] / (coal + 0.1)
sim_row["burning_zone_temp_C_roll1h"] = bz_temp
sim_row["kiln_torque_pct_roll1h"] = torque

X_sim = pd.DataFrame([sim_row[FEATURES]])
pred_free_lime = free_lime_model.predict(X_sim)[0]
pred_nox_risk = nox_model.predict_proba(X_sim)[0][1]

r1, r2 = st.columns(2)
with r1:
    st.metric("Predicted Free Lime %", f"{pred_free_lime:.2f} %",
              delta=f"{pred_free_lime - 1.2:.2f} vs target 1.2%", delta_color="inverse")
    if pred_free_lime > 1.8:
        st.warning("⬆️ High free lime risk — consider increasing burning zone temp or reducing feed rate.")
    elif pred_free_lime < 0.6:
        st.warning("⬇️ Over-burning risk — consider reducing fuel input to save energy.")
    else:
        st.success("✅ Free lime within optimal quality band.")

with r2:
    st.metric("Predicted NOx Breach Risk", f"{pred_nox_risk*100:.1f} %")
    if pred_nox_risk > 0.3:
        st.error("🚨 High NOx breach risk — recommend increasing AFR substitution or reducing burning zone temp.")
    else:
        st.success("✅ NOx risk within safe limits.")

st.divider()
st.caption("Built as a fully local, open-source automation pipeline: synthetic multi-source data → automated fusion pipeline → ML models → live control-recommendation dashboard. Ready to swap synthetic data for real DCS/SCADA feed.")
