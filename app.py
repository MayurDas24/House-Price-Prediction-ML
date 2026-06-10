
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Saved Model
# ==========================================

model = joblib.load("models/model.pkl")

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
    }

    .main {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# Title
# ==========================================

st.title("🏠 California House Price Prediction")

st.markdown("""
Predict California housing prices using an XGBoost Machine Learning model.
""")

st.divider()

# ==========================================
# User Inputs
# ==========================================

longitude = st.slider(
    "Longitude",
    min_value=-124.0,
    max_value=-114.0,
    value=-122.23
)

latitude = st.slider(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=37.88
)

housing_median_age = st.slider(
    "Housing Median Age",
    min_value=1,
    max_value=60,
    value=20
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1,
    value=1000
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1,
    value=200
)

population = st.number_input(
    "Population",
    min_value=1,
    value=500
)

households = st.number_input(
    "Households",
    min_value=1,
    value=300
)

median_income = st.slider(
    "Median Income",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

st.divider()

# ==========================================
# Prediction Button
# ==========================================

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ${prediction:,.2f}"
    )

    st.balloons()

# ==========================================
# Footer
# ==========================================

st.divider()

st.markdown("""
### ML Model Details
- Model: XGBoost Regressor
- R² Score: 0.825
- Dataset: California Housing Dataset
""")

