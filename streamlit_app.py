import os
# ---  SYSTEM CHECK & SILENCE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import requests
import urllib3
from sklearn.preprocessing import MinMaxScaler
import psycopg2

# 🚀 JAX & DEEPMIND LIBRARIES
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="Manchester Rail Forecast", layout="wide")
st.title("🚉 Manchester to Euston JAX Forecast")

# --- 2. LIVE API INTEGRATION ---
api_key = os.environ.get("RAIL_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("RAIL_API_KEY")
    except:
        api_key = None

manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']
headers = {'x-apikey': api_key, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

total_line_delay_mins = 0
total_line_delayed_trains = 0

if api_key:
    st.subheader("📡 Live Route Status")
    status_cols = st.columns(len(manchester_line))
    
    for idx, station in enumerate(manchester_line):
        current_api_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        api_url = f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepBoardWithDetails/{station}/{current_api_time}?numRows=10&timeWindow=120"
        
        station_delay = 0
        station_status = "🟢"
        
        try:
            response = requests.get(api_url, headers=headers, timeout=10, verify=False)
            if response.status_code == 200:
                live_data = response.json()
                trainServices = live_data.get('trainServices', [])
                
                for train in trainServices[:6]:
                    std = train.get('std', 'N/A')
                    etd = train.get('etd', 'N/A')
                    atd = train.get('atd', 'N/A')
                    is_cancelled = train.get('isCancelled', False)
                    current_time_flag = atd if atd != 'N/A' else etd
                    flag_lower = str(current_time_flag).lower().strip()
                    
                    if is_cancelled or flag_lower == 'cancelled':
                        station_delay += 60
                        total_line_delayed_trains += 1
                    elif flag_lower == 'delayed':
                        total_line_delayed_trains += 1
                    elif flag_lower not in ['on time', 'n/a', 'no report'] and std != 'N/A':
                        try:
                            try:
                                dt_std = datetime.fromisoformat(std)
                                dt_flag = datetime.fromisoformat(current_time_flag)
                            except ValueError:
                                dt_std = datetime.strptime(str(std).strip()[:5], '%H:%M')
                                dt_flag = datetime.strptime(str(current_time_flag).strip()[:5], '%H:%M')
                            delay = (dt_flag - dt_std).total_seconds() / 60
                            if delay < -720: delay += 1440
                            elif delay > 720: delay -= 1440
                            if delay > 0:
                                station_delay += delay
                                total_line_delayed_trains += 1
                        except:
                            pass
                
                total_line_delay_mins += station_delay
                if station_delay > 0:
                    station_status = f"🔴 {int(station_delay)}m"
                else:
                    station_status = "🟢 OK"
            else:
                station_status = "⚠️"
        except:
            station_status = "⚠️"
        
        status_cols[idx].metric(station, station_status)
        time.sleep(0.5)
    
    st.info(f"📊 Live snapshot: {total_line_delayed_trains} trains delayed/cancelled, ~{int(total_line_delay_mins)} minutes total delay")
else:
    st.warning("⚠️ RAIL_API_KEY not found. Skipping live status.")

# --- 3. LOAD DATA FROM DATABASE ---
st.divider()

@st.cache_data(show_spinner="Loading historical data from database...", ttl=3600)
def load_data():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        try:
            db_url = st.secrets.get("DATABASE_URL")
        except:
            db_url = None
    if not db_url:
        st.error("❌ DATABASE_URL not configured.")
        st.stop()
    
    conn = psycopg2.connect(db_url)
    df = pd.read_sql("""
        SELECT event_datetime as "EVENT_DATETIME", 
               pfpi_minutes as "PFPI_MINUTES",
               non_pfpi_minutes as "NON_PFPI_MINUTES"
        FROM rail_events 
        ORDER BY event_datetime
    """, conn)
    conn.close()
    return df

try:
    df = load_data()
    if df.empty:
        st.error("❌ No data in database. Run upload_to_db.py first.")
        st.stop()
    st.sidebar.success(f"✅ Loaded {len(df):,} rows from database.")
except Exception as e:
    st.error(f"❌ Database Error: {e}")
    st.stop()

# --- 4. DATA PROCESSING ---
df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], errors='coerce')
df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')

if 'NON_PFPI_MINUTES' not in df.columns:
    df['NON_PFPI_MINUTES'] = 0
if 'PFPI_MINUTES' not in df.columns:
    df['PFPI_MINUTES'] = 0

daily_performance = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES', 'NON_PFPI_MINUTES']].sum().reset_index()
daily_performance['TOTAL_COMBINED_MINUTES'] = daily_performance['PFPI_MINUTES'] + daily_performance['NON_PFPI_MINUTES']
daily_performance['EVENT_DATETIME'] = pd.to_datetime(daily_performance['EVENT_DATETIME'])

# --- LIVE API INJECTION ---
today_date_pd = pd.to_datetime(datetime.now().date())

if today_date_pd in daily_performance['EVENT_DATETIME'].values:
    idx = daily_performance.index[daily_performance['EVENT_DATETIME'] == today_date_pd][0]
    daily_performance.at[idx, 'TOTAL_COMBINED_MINUTES'] += total_line_delay_mins
else:
    new_row = pd.DataFrame({
        'EVENT_DATETIME': [today_date_pd],
        'PFPI_MINUTES': [total_line_delay_mins],
        'NON_PFPI_MINUTES': [0],
        'TOTAL_COMBINED_MINUTES': [total_line_delay_mins]
    })
    daily_performance = pd.concat([daily_performance, new_row], ignore_index=True)
    daily_performance = daily_performance.sort_values('EVENT_DATETIME').reset_index(drop=True)

total_days = len(daily_performance)

if total_days < 15:
    st.error(f"❌ Not enough data for training. Need at least 15 days, got {total_days}.")
    st.stop()

st.sidebar.info(f"📅 Data range: {daily_performance['EVENT_DATETIME'].min().date()} to {daily_performance['EVENT_DATETIME'].max().date()}")
st.sidebar.info(f"📊 Total days: {total_days}")

# --- 5. JAX MODEL TRAINING ---
st.subheader("🧠 Training JAX Predictive Model")

target_data = daily_performance.filter(['TOTAL_COMBINED_MINUTES'])
dataset = target_data.values
training_data_len = math.ceil(total_days * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

window_size = 14
train_data = scaled_data[0:training_data_len, :]

x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train_jnp = jnp.array(x_train).reshape(-1, window_size, 1)
y_train_jnp = jnp.array(y_train).reshape(-1, 1)

class JaxRouteModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

model = JaxRouteModel()
key = random.PRNGKey(42)
variables = model.init(key, jnp.ones((1, window_size, 1)))
params = variables['params']
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    def loss_fn(p):
        preds = model.apply({'params': p}, x)
        return jnp.mean((preds - y) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

progress_bar = st.progress(0)
status_text = st.empty()

for epoch in range(51):
    params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
    if epoch % 10 == 0:
        loss.block_until_ready()
        progress_bar.progress(epoch / 50)
        status_text.text(f"Epoch {epoch:02d} | Loss: {loss:.6f}")

progress_bar.progress(1.0)

# --- 6. TEST SET EVALUATION ---
test_data = scaled_data[training_data_len - window_size:, :]
x_test = [test_data[i - window_size:i, 0] for i in range(window_size, len(test_data))]
x_test_jnp = jnp.array(x_test).reshape(-1, window_size, 1)

y_test = dataset[training_data_len:, :]
preds_scaled = model.apply({'params': params}, x_test_jnp)
test_predictions = scaler.inverse_transform(np.array(preds_scaled))

rmse = np.sqrt(np.mean(((test_predictions - y_test) ** 2)))
st.success(f"✅ Model trained! Route RMSE: {rmse:.2f} Minutes")

# --- 7. FUTURE FORECAST ---
future_days = 30
future_predictions = []
current_window = scaled_data[-window_size:]

for i in range(future_days):
    in_win = jnp.reshape(current_window, (1, window_size, 1))
    next_pred = model.apply({'params': params}, in_win)
    future_predictions.append(next_pred[0, 0])
    current_window = jnp.append(current_window[1:], next_pred, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

valid_plot = daily_performance[training_data_len:].copy()
valid_plot['Predictions'] = test_predictions

last_actual_date = daily_performance['EVENT_DATETIME'].iloc[-1]
future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=future_days)
zoom_start_date = last_actual_date - pd.Timedelta(days=30)
zoomed_actuals = valid_plot[valid_plot['EVENT_DATETIME'] >= zoom_start_date]

# --- 8. VISUALIZATION ---
st.subheader("📈 30-Day Forecast")

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_title('Manchester to Euston: JAX Model 30-Day Forecast', fontsize=16, fontweight='bold')

ax.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['TOTAL_COMBINED_MINUTES'],
        label='Actual Delay Minutes (Last 30 Days)', color='darkorange', linewidth=2.5, marker='o')

ax.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['Predictions'],
        label='JAX Retrospective Test (Accuracy Check)', color='navy', linestyle='--', linewidth=2)

ax.plot(future_dates, future_predictions,
        label='Future JAX Forecast (Next 30 Days)', color='crimson', linestyle='-.', linewidth=2.5, marker='x')

ax.axvline(x=last_actual_date, color='black', linestyle='-', linewidth=2, label='Today (Data Cutoff)')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Route Delay Minutes', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)
fig.tight_layout()
st.pyplot(fig)

# --- 9. DAILY STRESS COEFFICIENT ---
st.subheader("📊 Route Stress Coefficient")

historical_daily = daily_performance[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
historical_daily.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL_COMBINED_MINUTES': 'Minutes'}, inplace=True)
historical_daily['Data_Type'] = 'Actual'

future_daily = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()})
future_daily['Data_Type'] = 'Forecast'

master_daily = pd.concat([historical_daily, future_daily], ignore_index=True)
master_daily['Stress_Coefficient'] = (master_daily['Minutes'] / historical_daily['Minutes'].max()).clip(upper=1.0)

st.dataframe(master_daily.tail(30), use_container_width=True)

st.balloons()