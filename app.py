
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_crop_yield_data.csv")
    metrics = pd.read_csv("model_metrics.csv")
    return df, metrics

model = load_model()
df, metrics = load_data()

st.title("🌱 Crop Yield Prediction System")
st.write("Sowing the Seeds of AI in Agriculture")
st.info("Enter agricultural details to estimate crop yield.")

st.subheader("Enter Crop Details")

state = st.selectbox(
    "Select State",
    sorted(df["State"].unique())
)

districts = sorted(
    df[df["State"] == state]["District"].unique()
)

district = st.selectbox("Select District", districts)

filtered = df[
    (df["State"] == state) &
    (df["District"] == district)
]

crops = sorted(filtered["Crop"].unique())
crop = st.selectbox("Select Crop", crops)

filtered_crop = filtered[filtered["Crop"] == crop]

seasons = sorted(filtered_crop["Season"].unique())
season = st.selectbox("Select Season", seasons)

area = st.number_input(
    "Cultivated Area (Lakh Hectares)",
    min_value=0.001,
    value=0.10,
    step=0.01
)

if st.button("🌾 Predict Crop Yield"):

    input_data = pd.DataFrame([{
        "State": state,
        "District": district,
        "Crop": crop,
        "Season": season,
        "Area": area
    }])

    prediction = float(model.predict(input_data)[0])

    st.success("Prediction Completed!")

    st.metric(
        "Predicted Crop Yield",
        f"{prediction:,.2f} kg/hectare"
    )

    st.metric(
        "Yield in Tonnes/Hectare",
        f"{prediction / 1000:.3f}"
    )

st.divider()

st.subheader("📊 Model Performance")
st.dataframe(metrics, hide_index=True)

st.subheader("🌱 Average Yield by Crop")

crop_summary = df.groupby("Crop")["Yield"].mean()
st.bar_chart(crop_summary)

st.caption(
    "Model trained using India's 2024–25 district-level dataset. "
    "Predictions are estimates, not guaranteed agricultural outcomes."
)
