"""
Streamlit app: GTD Attack Type Predictor
==========================================
Loads the trained Random Forest model + encoders (from src/train_model.py)
and predicts the likely attack type given historical/contextual features.

Run locally:   streamlit run src/app.py
Deploy:        push repo to GitHub, then deploy free on share.streamlit.io
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------------------------------------
# CONFIG / LOAD ARTIFACTS
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Attack Pattern Predictor", page_icon="📊", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return model, encoders, feature_cols


model, encoders, feature_cols = load_artifacts()
target_encoder = encoders["__target__"]

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("📊 Attack Pattern Predictor")
st.caption(
    "Predicts the likely **attack type** from historical event characteristics, "
    "trained on the Global Terrorism Database (170K+ real incident records, 1970–2016). "
    "This models aggregate event patterns — it does not identify or track any individual."
)

st.divider()

# -----------------------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------------------
st.subheader("Event characteristics")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", sorted(encoders["country_txt"].classes_))
    region = st.selectbox("Region", sorted(encoders["region_txt"].classes_))
    target_type = st.selectbox("Target type", sorted(encoders["Target"].classes_))
    weapon_type = st.selectbox("Weapon type", sorted(encoders["Weapons"].classes_))

with col2:
    group = st.selectbox(
        "Group / organization",
        sorted(encoders["Terrorist Group"].classes_),
        index=list(sorted(encoders["Terrorist Group"].classes_)).index("Unknown")
        if "Unknown" in encoders["Terrorist Group"].classes_ else 0,
    )
    year = st.number_input("Year", min_value=1970, max_value=2030, value=2016)
    month = st.number_input("Month", min_value=0, max_value=12, value=6)
    success = st.selectbox("Was it reported successful?", ["Successful", "Failed"])

col3, col4 = st.columns(2)
with col3:
    fatalities = st.number_input("Fatalities", min_value=0, value=0)
with col4:
    injured = st.number_input("Injured", min_value=0, value=0)

st.divider()

# -----------------------------------------------------------------------
# PREDICT
# -----------------------------------------------------------------------
if st.button("Predict attack type", type="primary", use_container_width=True):
    row = {
        "country_txt_enc": encoders["country_txt"].transform([country])[0],
        "region_txt_enc": encoders["region_txt"].transform([region])[0],
        "Target_enc": encoders["Target"].transform([target_type])[0],
        "Weapons_enc": encoders["Weapons"].transform([weapon_type])[0],
        "Terrorist Group_enc": encoders["Terrorist Group"].transform([group])[0],
        "iyear": year,
        "imonth": month,
        "Fatalities": fatalities,
        "Injured": injured,
        "Success_enc": 1 if success == "Successful" else 0,
    }
    X_input = pd.DataFrame([row])[feature_cols]

    pred_enc = model.predict(X_input)[0]
    pred_label = target_encoder.inverse_transform([pred_enc])[0]
    probs = model.predict_proba(X_input)[0]

    st.success(f"**Predicted attack type: {pred_label}**")

    prob_df = pd.DataFrame({
        "Attack Type": target_encoder.classes_,
        "Probability": probs,
    }).sort_values("Probability", ascending=False).reset_index(drop=True)

    st.subheader("Prediction confidence")
    st.bar_chart(prob_df.set_index("Attack Type")["Probability"])
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Dataset: Global Terrorism Database (START, University of Maryland). "
    "Model: Random Forest, tuned via GridSearchCV, cross-validated. "
    "Built for educational/portfolio purposes — not a real-time threat detection system."
)
