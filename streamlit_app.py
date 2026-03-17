import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import urllib3

# 🚀 JAX & DEEPMIND LIBRARIES
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Manchester Rail Forecast", layout="wide")
st.title("🚉 Manchester to Euston JAX Forecast")

# --- 2. DOWNLOAD DATA FROM GOOGLE DRIVE ---
FOLDER_ID = '1tzkHiffcIa-ZG_Evh3cY6huA-4fx4p2e'
DATA_DIR = 'drive_data'

@st.cache_resource(show_spinner="Downloading data from Google Drive...")
def download_folder():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # This downloads the entire folder content into DATA_DIR
    url = f'https://drive.google.com/drive/folders/{FOLDER_ID}'
    gdown.download_folder(url, output=DATA_DIR, quiet=True, use_cookies=False)
    return DATA_DIR

try:
    path = download_folder()
    # Find all CSVs in the downloaded folder
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        st.error("No CSV files found in the Drive folder. Check folder permissions!")
        st.stop()
    
    # Load and combine all CSVs
    df_list = [pd.read_csv(f, low_memory=False) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    st.sidebar.success(f"✅ Loaded {len(df):,} rows from {len(all_files)} files.")
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

# --- 3. LIVE RAIL API (Same as before) ---
try:
    api_key = st.secrets["RAIL_API_KEY"]
except:
    st.warning("⚠️ RAIL_API_KEY not found in Secrets. Skipping live status.")
    api_key = None

if api_key:
    st.subheader("📡 Live Route Status")
    manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']
    cols = st.columns(len(manchester_line))
    # (Simplified live check for display speed)
    for i, station in enumerate(manchester_line):
        cols[i].metric(station, "Scanning...")

# --- 4. DATA PROCESSING & JAX MODEL ---
st.divider()
st.subheader("🧠 Training JAX Predictive Model")

# Preprocessing
df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'].astype(str).str[:10], errors='coerce')
df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')
daily = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES']].sum().reset_index()

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