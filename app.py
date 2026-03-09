import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import config
from data_preprocessing import prepare_pipeline
from model_hawkes import SpatiotemporalHawkes
import os
import json
from datetime import datetime, timezone

st.set_page_config(page_title="Rocket Alert Predictor", layout="wide")

# --- PARAMETER PERSISTENCE LOGIC ---
PARAMS_FILE = "model_params.json"


def save_params(params):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f)


def load_params():
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Initial load from disk or config defaults
saved_data = load_params()

if 'mu' not in st.session_state:
    st.session_state.mu = saved_data['mu'] if saved_data else config.BACKGROUND_RATE
if 'alpha' not in st.session_state:
    st.session_state.alpha = saved_data['alpha'] if saved_data else config.TRIGGER_WEIGHT
if 'beta' not in st.session_state:
    st.session_state.beta = saved_data['beta'] if saved_data else config.DECAY_RATE_TIME
if 'sigma' not in st.session_state:
    st.session_state.sigma = saved_data['sigma'] if saved_data else config.DECAY_RATE_SPACE
if 'is_learning' not in st.session_state:
    st.session_state.is_learning = False


@st.cache_data
def load_data():
    processed_path = config.PROCESSED_DATA_DIR / "cleaned_alerts.csv"
    if not os.path.exists(processed_path):
        with st.spinner("Processing raw data..."):
            df = prepare_pipeline()
    else:
        df = pd.read_csv(processed_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


@st.cache_data
def load_cities():
    if os.path.exists(config.CITIES_CSV):
        return pd.read_csv(config.CITIES_CSV)
    return pd.DataFrame(columns=['City', 'Latitude', 'Longitude'])


df = load_data()
cities_df = load_cities()

# --- Initialize Model ---
model = SpatiotemporalHawkes(
    mu=st.session_state.mu,
    alpha=st.session_state.alpha,
    beta=st.session_state.beta,
    sigma=st.session_state.sigma
)
model.fit(df)

# --- UI Layout ---
st.title("🚀 Spatiotemporal Rocket Alert Predictor")

col1, col2 = st.columns([3, 1])

with col2:
    st.header("Time Control")
    min_dt, max_dt = df['timestamp'].min(), df['timestamp'].max()

    col_d, col_t = st.columns(2)
    with col_d:
        selected_date = st.date_input("Date", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    with col_t:
        selected_time = st.time_input("Time", value=max_dt.time())

    selected_dt = datetime.combine(selected_date, selected_time, tzinfo=timezone.utc)
    current_time_sec = (selected_dt - min_dt).total_seconds()

    st.subheader("Recent Alerts")
    st.dataframe(df[df['time_sec'] <= current_time_sec].tail(5)[['timestamp', 'zone']])

    st.header("🎯 Target Prediction")
    if not cities_df.empty:
        city_list = cities_df['City'].dropna().sort_values().tolist()
        selected_city = st.selectbox("Select a City:", city_list)

        if st.button("🔮 Predict Risk", use_container_width=True):
            city_data = cities_df[cities_df['City'] == selected_city].iloc[0]
            intensity = model.calculate_intensity(current_time_sec, city_data['Latitude'], city_data['Longitude'])

            max_val = max(st.session_state.mu * 10, 1e-6)
            risk_pct = min(100.0, (intensity / max_val) * 100)
            color = "#00cc66" if risk_pct < 15 else "#ffaa00" if risk_pct < 50 else "#ff3333"

            st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:rgba(255,255,255,0.1);'>"
                        f"<h4>{selected_city}</h4>"
                        f"<p>Risk Probability: <b style='font-size:1.2em; color:{color}'>{risk_pct:.1f}%</b></p></div>",
                        unsafe_allow_html=True)

    st.header("Model Parameters")
    if st.button("🧠 Learn & Save Parameters", use_container_width=True):
        st.session_state.is_learning = True

    if st.session_state.is_learning:
        with st.spinner("Optimizing..."):
            learned = model.learn_parameters(df[df['time_sec'] <= current_time_sec])
            # Update state
            st.session_state.mu, st.session_state.alpha, st.session_state.beta = learned['mu'], learned['alpha'], \
            learned['beta']
            # Save to local disk!
            save_params({'mu': learned['mu'], 'alpha': learned['alpha'], 'beta': learned['beta'],
                         'sigma': st.session_state.sigma})
            st.session_state.is_learning = False
            st.success("Parameters saved to disk!")
            st.rerun()

    st.session_state.alpha = st.slider("Trigger Weight (Alpha)", 0.0, 1.0, float(st.session_state.alpha))
    st.session_state.beta = st.slider("Temporal Decay (Beta)", 0.001, 2.0, float(st.session_state.beta))
    st.session_state.sigma = st.slider("Spatial Spread (Sigma)", 0.01, 1.0, float(st.session_state.sigma))

with col1:
    with st.spinner("Updating Map..."):
        city_risks = []
        max_val = max(st.session_state.mu * 10, 1e-6)
        for _, row in cities_df.dropna(subset=['Latitude', 'Longitude']).iterrows():
            intensity = model.calculate_intensity(current_time_sec, row['Latitude'], row['Longitude'])
            p = min(100.0, (intensity / max_val) * 100) / 100.0
            color = [int(255 * min(1, 2 * p)), int(255 * min(1, 2 * (1 - p))), 0, 160]
            city_risks.append({"City": row['City'], "latitude": row['Latitude'], "longitude": row['Longitude'],
                               "risk": f"{p * 100:.1f}%", "color": color})

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame(city_risks), get_position=["longitude", "latitude"],
                          get_fill_color="color", get_radius=3000, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=config.MAP_CENTER_LAT, longitude=config.MAP_CENTER_LON,
                                         zoom=config.DEFAULT_ZOOM),
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "City: {City}\nRisk: {risk}"}
    ))
