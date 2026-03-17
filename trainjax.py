import os
# ---  SYSTEM CHECK & SILENCE WARNINGS ---
# THIS MUST BE AT THE VERY TOP BEFORE ANY JAX IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['JAX_PLATFORMS'] = 'cpu' 

import streamlit as st
import pandas as pd
import numpy as np
import glob
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import requests
import urllib3
from sklearn.preprocessing import MinMaxScaler

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

# ---  LIVE API INTEGRATION ---
print("📡 --- SCANNING LIVE DEPARTURES (MAN to EUS) ---")

api_key = "EhPYIKPzBrWdoIqeA6u1hGc54eJSCcZxiGGgGqfGSwkwuGVQ"
manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS'] 
headers = {'x-apikey': api_key, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

total_line_delay_mins = 0
total_line_delayed_trains = 0

for station in manchester_line:
    current_api_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    api_url = f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepBoardWithDetails/{station}/{current_api_time}?numRows=10&timeWindow=120"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            live_data = response.json()
            trainServices = live_data.get('trainServices', [])
            
            station_delay_mins, station_delayed_trains = 0, 0
            print(f"\n📍 {station} DEPARTURES:")
            
            if not trainServices:
                print("  ↳ No trains scheduled in the next 120 minutes.")
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
        else:
            print(f"⚠️ {station}: API Error {response.status_code}")
    except Exception as e:
         print(f"⚠️ {station}: Connection Failed")
    
    time.sleep(1)

print("-" * 55)
print(f"📊 LIVE ROUTE SNAPSHOT: {total_line_delayed_trains} trains delayed/cancelled on the mainline, totaling ~{total_line_delay_mins} minutes right now.\n")

base_path = r"/mnt/c/Users/OWNER/OneDrive - University of Keele/Disitation/code" 
all_files = glob.glob(os.path.join(base_path, "data*.csv"))

if not all_files:
    base_path = r"C:\Users\OWNER\OneDrive - University of Keele\Disitation\code"
    all_files = glob.glob(os.path.join(base_path, "data*.csv"))
    if not all_files:
        print("❌ Error: No data files found. Check your folder path.")
        exit()

print(f"📂 Found {len(all_files)} files. Checking for corrupted/empty files...")

df_list = []
for f in all_files:
    try:
        if os.path.getsize(f) > 0:
            temp_df = pd.read_csv(f, low_memory=False)
            df_list.append(temp_df)
    except Exception:
        pass

if not df_list:
    print("❌ CRITICAL ERROR: All found CSV files were empty or corrupted!")
    exit()

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Loaded {len(df)} rows from {len(df_list)} files.")

station_col = None
for col in df.columns:
    if str(col).strip().upper() in ['CRS', 'STATION', 'LOCATION', 'TPL', 'STATION_CODE']:
        station_col = col
        break

if station_col:
    df[station_col] = df[station_col].astype(str).str.strip().str.upper()
    df = df[df[station_col].isin(manchester_line)] 
    print(f" Filtered historical data down to route stations only using column: '{station_col}'.")
else:
    print("⚠️ WARNING: Could not find ANY station column (like 'CRS' or 'tpl'). Training on ALL data instead.")

df['DATE_STRING'] = df['EVENT_DATETIME'].astype(str).str[:10]
df['EVENT_DATETIME'] = pd.to_datetime(df['DATE_STRING'], errors='coerce', format='mixed', dayfirst=True)
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

print(f"\n🧠 --- TRAINING JAX AI ON MANCHESTER-EUSTON ROUTE ---")
for epoch in range(51): 
    start_t = time.time()
    params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
    if epoch % 10 == 0:
        loss.block_until_ready() 
        dur = (time.time() - start_t) * 1000
        print(f"Epoch {epoch:02d} | Loss: {loss:.6f} | Compute Speed: {dur:.2f}ms")

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

plt.figure(figsize=(16, 7))
plt.title(f'Manchester to Euston: JAX Model 30-Day Forecast', fontsize=16, fontweight='bold')

plt.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['TOTAL_COMBINED_MINUTES'], 
         label='Actual Delay Minutes (Last 30 Days)', color='darkorange', linewidth=2.5, marker='o')

plt.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['Predictions'], 
         label='JAX Retrospective Test (Accuracy Check)', color='navy', linestyle='--', linewidth=2)

plt.plot(future_dates, future_predictions, 
         label='Future JAX Forecast (Next 30 Days)', color='crimson', linestyle='-.', linewidth=2.5, marker='x')

plt.axvline(x=last_actual_date, color='black', linestyle='-', linewidth=2, label='Today (Data Cutoff)')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3)) 
plt.xticks(rotation=45)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Route Delay Minutes', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

chart_path = os.path.join(base_path, "Manchester_to_Euston_JAX_Forecast.png")
plt.savefig(chart_path, dpi=300)
print(f" Route Chart auto-updated: {chart_path}")

# ---  DAILY ROUTE COEFFICIENT ---
historical_daily = daily_performance[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
historical_daily.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL_COMBINED_MINUTES': 'Minutes'}, inplace=True)
historical_daily['Data_Type'] = 'Actual'

future_daily = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()})
future_daily['Data_Type'] = 'Forecast'

master_daily = pd.concat([historical_daily, future_daily], ignore_index=True)
master_daily['Stress_Coefficient'] = (master_daily['Minutes'] / historical_daily['Minutes'].max()).clip(upper=1.0) 

csv_path = os.path.join(base_path, "jax_coeficient.csv")
master_daily.to_csv(csv_path, index=False)

print("\n🎉 JAX ROUTE PIPELINE COMPLETE!")