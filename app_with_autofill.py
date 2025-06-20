
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model components
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Metro Cost Prediction", layout="wide")
st.title("🚇 Metro Station Civil Cost Predictor (ML-Based)")
st.markdown("Predict estimated civil cost for metro stations using your trained machine learning model.")

st.sidebar.header("📥 Input Station Parameters")

# === Sample presets for autofill ===
presets = {
    "Regular Station": {
        "depth_m": 18, "platform_length_m": 140, "no_of_escalators": 6,
        "no_of_elevators": 2, "no_of_entry_exit_points": 2, "no_of_tracks": 2,
        "total_area_sqm": 15000
    },
    "Terminal Station": {
        "depth_m": 20, "platform_length_m": 160, "no_of_escalators": 8,
        "no_of_elevators": 3, "no_of_entry_exit_points": 3, "no_of_tracks": 3,
        "total_area_sqm": 18000
    },
    "Interchange Station": {
        "depth_m": 25, "platform_length_m": 180, "no_of_escalators": 10,
        "no_of_elevators": 4, "no_of_entry_exit_points": 4, "no_of_tracks": 4,
        "total_area_sqm": 22000
    }
}

# Select preset
preset_choice = st.sidebar.selectbox("🧠 Autofill with Station Type:", ["Custom Input"] + list(presets.keys()))

# Load reference ranges
try:
    ref_df = pd.read_excel("Metro_Data.xlsx")
    ref_df = ref_df.select_dtypes(include='number')
except:
    ref_df = pd.DataFrame(columns=features)

input_data = {}

# Display inputs
for col in features:
    min_val = float(ref_df[col].min()) if col in ref_df.columns else 0.0
    max_val = float(ref_df[col].max()) if col in ref_df.columns else 1000.0
    default = presets[preset_choice][col] if preset_choice != "Custom Input" and col in presets[preset_choice] else               float(ref_df[col].mean()) if col in ref_df.columns else (max_val + min_val) / 2
    step = (max_val - min_val) / 100 if max_val > min_val else 1.0
    input_data[col] = st.sidebar.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step,
        format="%.2f"
    )

# Prediction section
if st.button("🔮 Predict Civil Cost"):
    user_df = pd.DataFrame([input_data])
    user_imputed = imputer.transform(user_df)
    user_scaled = scaler.transform(user_imputed)
    prediction = model.predict(user_scaled)[0]
    st.success(f"💰 **Predicted Civil Cost:** ₹{prediction:.2f} Cr")

# Batch prediction upload
st.markdown("---")
st.subheader("📄 Upload Excel File for Batch Prediction")
uploaded_file = st.file_uploader("Upload an Excel file with station data", type=["xlsx"])

if uploaded_file:
    batch_df = pd.read_excel(uploaded_file)
    if all(col in batch_df.columns for col in features):
        imputed = imputer.transform(batch_df[features])
        scaled = scaler.transform(imputed)
        preds = model.predict(scaled)
        batch_df["Predicted_Civil_Cost_Cr"] = preds
        st.success("✅ Predictions completed.")
        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, file_name="predicted_costs.csv")
    else:
        st.error("❌ The uploaded file is missing some required feature columns.")
