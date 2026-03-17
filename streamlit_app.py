import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import urllib3
import psycopg2

# 🚀 JAX & DEEPMIND LIBRARIES
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Manchester Rail Forecast", layout="wide")
st.title("🚉 Manchester to Euston JAX Forecast")

# --- 2. LOAD DATA FROM DIGITAL OCEAN POSTGRESQL ---
@st.cache_data(show_spinner="Loading data from database...", ttl=3600)
def load_data():
    db_url = os.environ.get("DATABASE_URL") or st.secrets.get("DATABASE_URL")
    if not db_url:
        st.error("❌ DATABASE_URL not configured. Add it as an environment variable.")
        st.stop()
    
    conn = psycopg2.connect(db_url)
    df = pd.read_sql("SELECT event_datetime, pfpi_minutes FROM rail_events ORDER BY event_datetime", conn)
    conn.close()
    
    # Match original column names
    df.columns = ['EVENT_DATETIME', 'PFPI_MINUTES']
    return df

try:
    df = load_data()
    if df.empty:
        st.error("❌ No data found in database. Please run upload_to_db.py to load your CSV data.")
        st.stop()
    st.sidebar.success(f"✅ Loaded {len(df):,} rows from database.")
except Exception as e:
    st.error(f"❌ Database Connection Error: {e}")
    st.stop()

# --- 3. LIVE RAIL API ---
api_key = os.environ.get("RAIL_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("RAIL_API_KEY")
    except:
        api_key = None

if not api_key:
    st.warning("⚠️ RAIL_API_KEY not found. Skipping live status.")

if api_key:
    st.subheader("📡 Live Route Status")
    manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']
    cols = st.columns(len(manchester_line))
    for i, station in enumerate(manchester_line):
        cols[i].metric(station, "Scanning...")

# --- 4. DATA PROCESSING & JAX MODEL ---
st.divider()
st.subheader("🧠 Training JAX Predictive Model")

# Preprocessing
df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], errors='coerce')
df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')
daily = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES']].sum().reset_index()

if daily.empty:
    st.error("❌ No valid data after processing. Check your database contents.")
    st.stop()

# JAX Model Setup
class JaxRouteModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

window_size = 14
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily[['PFPI_MINUTES']].values)

# Initialize JAX
model = JaxRouteModel()
params = model.init(random.PRNGKey(0), jnp.ones((1, window_size, 1)))['params']

st.success(f"Model initialized. Data range: {daily['EVENT_DATETIME'].min()} to {daily['EVENT_DATETIME'].max()}")

# --- 5. VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily['EVENT_DATETIME'], daily['PFPI_MINUTES'], label='Delay Minutes', color='#1f77b4')
ax.set_title("Historical Delay Trends")
ax.set_ylabel("Total Minutes")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.balloons()