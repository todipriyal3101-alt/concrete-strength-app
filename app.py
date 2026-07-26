import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱", layout="centered")

st.title("🧱 Concrete Compressive Strength Predictor")
st.write(
    "Enter the mix composition and curing age to predict 28-day-style "
    "compressive strength (MPa) using a tuned XGBoost model."
)

# ---------- Load model (cached so it isn't reloaded on every interaction) ----------
@st.cache_resource
def load_model():
    return joblib.load("concrete_strength_model.pkl")

model = load_model()

# ---------- Input form ----------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input("Cement (kg/m³)", min_value=0.0, max_value=600.0, value=300.0, step=1.0)
        blast_furnace_slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0, value=0.0, step=1.0)
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
        water = st.number_input("Water (kg/m³)", min_value=0.0, max_value=300.0, value=180.0, step=1.0)

    with col2:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=35.0, value=5.0, step=0.5)
        coarse_aggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, max_value=1200.0, value=1000.0, step=1.0)
        fine_aggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, max_value=1000.0, value=700.0, step=1.0)
        age = st.number_input("Age (days)", min_value=1, max_value=365, value=28, step=1)

    submitted = st.form_submit_button("Predict Strength")

# ---------- Prediction ----------
if submitted:
    input_df = pd.DataFrame([{
        "cement": cement,
        "blast_furnace_slag": blast_furnace_slag,
        "fly_ash": fly_ash,
        "water": water,
        "superplasticizer": superplasticizer,
        "coarse_aggregate": coarse_aggregate,
        "fine_aggregate": fine_aggregate,
        "age": age
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Compressive Strength: **{prediction:.2f} MPa**")

    with st.expander("See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.markdown("---")
st.caption(
    "Model: Tuned XGBoost Regressor · Trained on the Yeh (1998) Concrete Compressive "
    "Strength dataset · Test R² ≈ 0.94"
)
