import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bridge Health Monitoring", layout="wide")
st.title("Structural Health Monitoring Dashboard")

# ===== Load saved model & scaler =====
model = joblib.load("semi_supervised_model.pkl")
scaler = joblib.load("scaler_bridge.pkl")
df = pd.read_csv("bridge_dataset.csv")


# ===== Features from training =====
features = [
    "temperature_c",
    "humidity_percent",
    "wind_speed_mps",
    "fft_peak_freq",
    "fft_magnitude",
    "degradation_score",
    "structural_condition",
    "forecast_score_next_30d"
]


# ===== Scaling & Prediction =====
X_scaled = scaler.transform(df[features])
predictions = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)

confidence_scores = np.max(probs, axis=1)
# ===== Results DataFrame =====
results = df.copy()
results["Prediction"] = predictions
results["Confidence"] = confidence_scores

# Keep original prediction if it's "Abnormal"
def label_with_conf(pred, conf, threshold=0.6):
    if conf < threshold and pred != "Abnormal":
        return "Maybe Abnormal"
    return pred

results["Confidence_Label"] = [
    label_with_conf(p, c) for p, c in zip(results["Prediction"], results["Confidence"])
]



# ===== Ambiguous vs Confident Predictions =====
st.subheader("Ambiguous vs Confident Predictions")
conf_counts = results["Confidence_Label"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(
    conf_counts,
    labels=conf_counts.index,
    autopct="%1.1f%%",
    colors=["green", "orange", "red"]
)
st.pyplot(fig1)



# ===== Signal Trend with Predictions =====
st.subheader("Signal Trend with Predictions")
if "signal" not in df.columns:
    if {"acceleration_x", "acceleration_y", "acceleration_z"}.issubset(df.columns):
        df["signal"] = np.sqrt(
            df["acceleration_x"]**2 +
            df["acceleration_y"]**2 +
            df["acceleration_z"]**2
        )
    else:
        st.warning("No acceleration columns found to generate signal trend.")
        df["signal"] = None

if df["signal"].notnull().any():
    plt.figure(figsize=(10, 5))
    plt.plot(df["signal"], label="Signal", color="blue", alpha=0.5)
    colors = {"Normal": "green", "Maybe Abnormal": "orange", "Abnormal": "red"}
    for i, label in enumerate(results["Confidence_Label"]):
        plt.scatter(i, df["signal"].iloc[i], color=colors[label], s=10)
    plt.legend()
    st.pyplot(plt)
else:
    st.warning("Signal data not available for plotting.")

# ===== Real-time “Maybe Abnormal” Alerts =====
st.subheader("Real-time Alerts: Maybe Abnormal Cases")
if "timestamp" in results.columns:
    alerts = results[results["Confidence_Label"] == "Maybe Abnormal"][["timestamp", "Confidence"]]
    if not alerts.empty:
        st.dataframe(alerts)
    else:
        st.success("No 'Maybe Abnormal' alerts at the moment.")
else:
    st.warning("Timestamp column not found for alerts.")
