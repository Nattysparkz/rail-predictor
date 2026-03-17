import os
# ---  SYSTEM CHECK & SILENCE WARNINGS ---
# THIS MUST BE AT THE VERY TOP BEFORE ANY JAX IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['JAX_PLATFORMS'] = 'cpu' 

import streamlit as st
st.set_page_config(page_title="Manchester Rail Forecast", layout="wide")

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

current_time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"=======================================================")
print(f"🚀 JAX/FLAX PIPELINE: MANCHESTER TO EUSTON ROUTE")
print(f"🕒 System Time: {current_time_display}")
print(f"=======================================================\n")

st.title("🚉 Manchester to Euston JAX Forecast")

# ---  LIVE API INTEGRATION ---
print("📡 --- SCANNING LIVE DEPARTURES (MAN to EUS) ---")

api_key = os.environ.get("RAIL_API_KEY", "EhPYIKPzBrWdoIqeA6u1hGc54eJSCcZxiGGgGqfGSwkwuGVQ")
manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS'] 
consumer_secret = os.environ.get("RAIL_API_SECRET", "21D2LAgmc11iBFNQFpjzmspElbkXTEMapngoc74Guutj9NZqdBoz8574DQWVkkrg")
headers = {'x-apikey': api_key, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

total_line_delay_mins = 0
total_line_delayed_trains = 0
api_available = False
live_results = []
api_debug_log = []
raw_env_key = os.environ.get('RAIL_API_KEY', '')
api_debug_log.append(f"API Key source: {'ENV' if raw_env_key else 'HARDCODED'}")
api_debug_log.append(f"API Key length: {len(api_key)}")
api_debug_log.append(f"API Key preview: {api_key[:8]}...{api_key[-4:]}")
api_debug_log.append(f"Headers: {headers}")

for station in manchester_line:
    current_api_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    # Try detailed board first, then simpler endpoint as fallback
    api_urls = [
        f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepBoardWithDetails/{station}/{current_api_time}?numRows=10&timeWindow=120",
        f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepartureBoardByCRS/{station}/{current_api_time}?numRows=10&timeWindow=120",
    ]
    
    response = None
    for url in api_urls:
        endpoint_name = url.split('/')[-3]
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=True)
            if response.status_code == 200:
                api_debug_log.append(f"✅ {station}: {endpoint_name} → 200 OK")
                break
            api_debug_log.append(f"⚠️ {station}: {endpoint_name} → {response.status_code}")
            try:
                api_debug_log.append(f"   Response: {response.text[:200]}")
            except:
                pass
            print(f"⚠️ {station}: {endpoint_name} returned {response.status_code}")
        except requests.exceptions.SSLError as e:
            api_debug_log.append(f"🔒 {station}: {endpoint_name} → SSL Error, retrying without verify...")
            try:
                response = requests.get(url, headers=headers, timeout=15, verify=False)
                if response.status_code == 200:
                    api_debug_log.append(f"✅ {station}: {endpoint_name} → 200 OK (no SSL verify)")
                    break
                api_debug_log.append(f"⚠️ {station}: {endpoint_name} → {response.status_code} (no SSL verify)")
            except Exception as e2:
                api_debug_log.append(f"❌ {station}: {endpoint_name} → SSL retry failed: {str(e2)[:80]}")
                continue
        except requests.exceptions.Timeout:
            api_debug_log.append(f"⏱️ {station}: {endpoint_name} → Timeout (15s)")
            continue
        except requests.exceptions.ConnectionError as e:
            api_debug_log.append(f"❌ {station}: {endpoint_name} → Connection Error: {str(e)[:80]}")
            continue
        except Exception as e:
            api_debug_log.append(f"❌ {station}: {endpoint_name} → {str(e)[:80]}")
            print(f"⚠️ {station}: {e}")
            continue
    
    try:
        if response and response.status_code == 200:
            api_available = True
            live_data = response.json()
            trainServices = live_data.get('trainServices', [])
            
            station_delay_mins, station_delayed_trains = 0, 0
            print(f"\n📍 {station} DEPARTURES:")
            
            if not trainServices:
                print("  ↳ No trains scheduled in the next 120 minutes.")
                live_results.append({'station': station, 'status': '⚪ None', 'delay': 0})
                continue

            for train in trainServices[:6]: 
                std = train.get('std', 'N/A')
                etd = train.get('etd', 'N/A')
                atd = train.get('atd', 'N/A') 
                is_cancelled = train.get('isCancelled', False)
                destination = train.get('destination', [{}])[0].get('locationName', 'Unknown')
                
                current_time_flag = atd if atd != 'N/A' else etd
                status = "🟢 On Time"
                flag_lower = str(current_time_flag).lower().strip()
                
                if is_cancelled or flag_lower == 'cancelled':
                    status = "❌ CANCELLED"
                    station_delay_mins += 60 
                    station_delayed_trains += 1
                elif flag_lower == 'delayed':
                    status = "🔴 Delayed (No ETD)"
                    station_delayed_trains += 1
                elif flag_lower == 'no report':
                    status = "🟡 No Report (Lost on radar)"
                elif flag_lower not in ['on time', 'n/a'] and std != 'N/A':
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
                            station_delay_mins += delay
                            station_delayed_trains += 1
                            status = f"🔴 {int(delay)} mins late"
                            if atd != 'N/A': status += " (Departed)"
                        elif delay <= 0 and atd != 'N/A':
                            status = "🟢 On Time (Departed)"
                            
                    except Exception as e:
                        status = f"🟡 Unknown Format: [{current_time_flag}]"
                
                try:
                    display_time = datetime.fromisoformat(std).strftime('%H:%M')
                except:
                    display_time = str(std)[:5]
                
                print(f"  ↳ To: {destination[:15]:<15} | Due: {display_time} | {status}")
                
            total_line_delay_mins += station_delay_mins
            total_line_delayed_trains += station_delayed_trains
            
            if station_delay_mins > 0:
                live_results.append({'station': station, 'status': f'🔴 {int(station_delay_mins)}m', 'delay': station_delay_mins})
            else:
                live_results.append({'station': station, 'status': '🟢 OK', 'delay': 0})
        else:
            status_code = response.status_code if response else "No response"
            print(f"⚠️ {station}: API Error {status_code}")
            live_results.append({'station': station, 'status': f'⚠️', 'delay': 0})
    except Exception as e:
         print(f"⚠️ {station}: Connection Failed")
         live_results.append({'station': station, 'status': '⚠️', 'delay': 0})
    
    time.sleep(1)

print("-" * 55)
print(f"📊 LIVE ROUTE SNAPSHOT: {total_line_delayed_trains} trains delayed/cancelled on the mainline, totaling ~{total_line_delay_mins} minutes right now.\n")

# --- DISPLAY LIVE STATUS IN STREAMLIT ---
st.subheader("📡 Live Route Status")

if api_available:
    status_cols = st.columns(len(manchester_line))
    for i, result in enumerate(live_results):
        status_cols[i].metric(result['station'], result['status'])
    st.info(f"📊 Live snapshot: {total_line_delayed_trains} trains delayed/cancelled, ~{int(total_line_delay_mins)} minutes total delay")
else:
    st.warning(
        "📡 **Live departure data is temporarily unavailable.** "
        "The Rail API may be unreachable from this server — this does not affect the forecast model below. "
        "Live data will appear automatically when the connection is restored. "
        f"Last checked: {datetime.now().strftime('%H:%M:%S %d %b %Y')}"
    )

# Show API debug info
with st.expander("🔧 API Connection Debug Log"):
    for log_line in api_debug_log:
        st.text(log_line)
    if not api_debug_log:
        st.text("No API requests were attempted.")

# --- LOAD DATA FROM DIGITAL OCEAN POSTGRESQL ---
print("📂 --- LOADING HISTORICAL DATA FROM DATABASE ---")

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    try:
        db_url = st.secrets.get("DATABASE_URL")
    except:
        db_url = None

if not db_url:
    print("❌ Error: DATABASE_URL not configured.")
    st.error("❌ DATABASE_URL not configured. Add it as an environment variable.")
    st.stop()

try:
    conn = psycopg2.connect(db_url)
    df = pd.read_sql("""
        SELECT event_datetime as "EVENT_DATETIME", 
               pfpi_minutes as "PFPI_MINUTES",
               non_pfpi_minutes as "NON_PFPI_MINUTES"
        FROM rail_events 
        ORDER BY event_datetime
    """, conn)
    conn.close()
    print(f"✅ Loaded {len(df)} rows from database.")
except Exception as e:
    print(f"❌ Database error: {e}")
    st.error(f"❌ Database Error: {e}")
    st.stop()

if df.empty:
    print("❌ CRITICAL ERROR: No data in database!")
    st.error("❌ No data in database. Run upload_to_db.py to load your CSV data first.")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(df):,} rows from database.")

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

st.sidebar.info(f"📅 Data: {daily_performance['EVENT_DATETIME'].min().date()} to {daily_performance['EVENT_DATETIME'].max().date()}")
st.sidebar.info(f"📊 Total days: {total_days}")

#
target_data = daily_performance.filter(['TOTAL_COMBINED_MINUTES'])
dataset = target_data.values
training_data_len = math.ceil(total_days * 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

window_size = 14 
train_data = scaled_data[0:training_data_len, :]

x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train_jnp = jnp.array(x_train).reshape(-1, window_size, 1)
y_train_jnp = jnp.array(y_train).reshape(-1, 1)

# ---  BUILDING THE JAX/FLAX AI ---
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

# ---  JAX ACCELERATED TRAINING ENGINE ---
@jax.jit
def train_step(params, opt_state, x, y):
    def loss_fn(p):
        preds = model.apply({'params': p}, x)
        return jnp.mean((preds - y) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

st.divider()
st.subheader("🧠 Training JAX Predictive Model")
progress_bar = st.progress(0)
status_text = st.empty()

print(f"\n🧠 --- TRAINING JAX AI ON MANCHESTER-EUSTON ROUTE ---")
for epoch in range(51): 
    start_t = time.time()
    params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
    if epoch % 10 == 0:
        loss.block_until_ready() 
        dur = (time.time() - start_t) * 1000
        print(f"Epoch {epoch:02d} | Loss: {loss:.6f} | Compute Speed: {dur:.2f}ms")
        progress_bar.progress(epoch / 50)
        status_text.text(f"Epoch {epoch:02d} | Loss: {loss:.6f} | Speed: {dur:.1f}ms")

progress_bar.progress(1.0)

# ---  TESTING THE AI (20% UNSEEN DATA) ---
print("\n🎯 --- EVALUATING JAX TEST SET (20% DATA) ---")
test_data = scaled_data[training_data_len - window_size: , :]
x_test = [test_data[i-window_size:i, 0] for i in range(window_size, len(test_data))]
x_test_jnp = jnp.array(x_test).reshape(-1, window_size, 1)

y_test = dataset[training_data_len:, :]
preds_scaled = model.apply({'params': params}, x_test_jnp)
test_predictions = scaler.inverse_transform(np.array(preds_scaled))

rmse = np.sqrt(np.mean(((test_predictions - y_test) ** 2)))
print(f" Route RMSE on Test Data: {rmse:.2f} Minutes")

st.success(f"✅ Model trained! Route RMSE: {rmse:.2f} Minutes")
st.sidebar.metric("RMSE", f"{rmse:.2f} mins")

# ---  FUTURE FORECAST  ---
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

# --- FORECAST CHART ---
st.subheader("📈 30-Day JAX Forecast")

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_title(f'Manchester to Euston: JAX Model 30-Day Forecast', fontsize=16, fontweight='bold')

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

# ---  DAILY ROUTE COEFFICIENT ---
st.subheader("📊 Route Stress Coefficient")

historical_daily = daily_performance[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
historical_daily.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL_COMBINED_MINUTES': 'Minutes'}, inplace=True)
historical_daily['Data_Type'] = 'Actual'

future_daily = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()})
future_daily['Data_Type'] = 'Forecast'

master_daily = pd.concat([historical_daily, future_daily], ignore_index=True)
master_daily['Stress_Coefficient'] = (master_daily['Minutes'] / historical_daily['Minutes'].max()).clip(upper=1.0) 

st.dataframe(master_daily.tail(30), use_container_width=True)

print("\n🎉 JAX ROUTE PIPELINE COMPLETE!")
st.balloons()