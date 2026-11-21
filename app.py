
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

st.set_page_config(
    page_title="Dengue Climate Risk Simulator",
    page_icon="🦟",
    layout="wide"
)

st.title("Dengue Climate Risk Simulator")
st.caption(
    "Makassar, Indonesia | Random Forest | "
    "Historical Data 2017–2021"
)

st.warning(
    "Research prototype only. This model has not "
    "been validated for clinical decisions or "
    "operational dengue early warnings."
)

@st.cache_resource
def load_model():
    model = joblib.load("dengue_random_forest.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
except (FileNotFoundError, OSError, ValueError) as e:
    st.error(
        "Model files are not available. "
        "Upload dengue_random_forest.pkl and "
        "model_metadata.json to this repository."
    )
    st.stop()

fitur = metadata["features"]

st.sidebar.header("Climate Variables")
st.sidebar.caption(
    "Enter meteorological conditions from "
    "two months before the target month."
)

rainfall = st.sidebar.slider(
    "Rainfall (mm)", 0.0, 950.0, 176.5, 0.5
)

rainy_days = st.sidebar.slider(
    "Rainy days", 0, 31, 15
)

humidity = st.sidebar.slider(
    "Humidity (%)", 50.0, 100.0, 79.5, 0.5
)

temperature = st.sidebar.slider(
    "Air temperature (°C)", 20.0, 35.0, 28.1, 0.1
)

sunshine = st.sidebar.slider(
    "Sunshine duration", 0.0, 100.0, 69.5, 0.5
)

input_data = pd.DataFrame([{
    "L2_Rainfall": rainfall,
    "L2_Rainy_Day": rainy_days,
    "L2_Humidity": humidity,
    "L2_Temp": temperature,
    "L2_Sunshine": sunshine
}])[fitur]

prediksi = model.predict(input_data)[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Estimated dengue cases",
    f"{prediksi:.1f}"
)

col2.metric(
    "Test MAE",
    f"{metadata['mae']:.2f}"
)

col3.metric(
    "Test RMSE",
    f"{metadata['rmse']:.2f}"
)

st.divider()

st.subheader("Rainfall Scenario Analysis")

rentang_hujan = np.linspace(0, 950, 40)

skenario = pd.DataFrame({
    "L2_Rainfall": rentang_hujan,
    "L2_Rainy_Day": rainy_days,
    "L2_Humidity": humidity,
    "L2_Temp": temperature,
    "L2_Sunshine": sunshine
})[fitur]

hasil = model.predict(skenario)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    rentang_hujan,
    hasil,
    marker="o",
    markersize=3
)

ax.axvline(
    rainfall,
    linestyle="--",
    label="Selected rainfall"
)

ax.set_xlabel("Rainfall two months earlier (mm)")
ax.set_ylabel("Estimated monthly dengue cases")
ax.set_title("Model Sensitivity to Rainfall")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)

st.subheader("Current Scenario Inputs")
st.dataframe(input_data, hide_index=True)

st.info(
    "Changing one climate variable while holding "
    "others constant is a model sensitivity exercise, "
    "not evidence of a causal relationship."
)

st.divider()

st.caption(
    "Model trained on Makassar 2017–2020 data and "
    "tested on 2021 data. Not validated for "
    "Konawe Kepulauan, Wakatobi, or other districts."
)
