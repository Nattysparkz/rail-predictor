import pandas as pd
import numpy as np
import os
import glob
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import requests
import urllib3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 0. AUTO-UPDATE SYSTEM CHECK ---
current_time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"=======================================================")
print(f"🔄 AI PIPELINE: MANCHESTER TO EUSTON ROUTE ANALYSIS")
print(f"🕒 System Time: {current_time_display}")
print(f"=======================================================\n")

# --- 1. LIVE API INTEGRATION (ENTIRE ROUTE) ---
print("📡 --- SCANNING LIVE DEPARTURES (MAN to EUS) ---")

api_key = "EhPYIKPzBrWdoIqeA6u1hGc54eJSCcZxiGGgGqfGSwkwuGVQ"
manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'RUG', 'MKC', 'EUS'] 
headers = {'x-apikey': api_key, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

total_line_delay_mins = 0
total_line_delayed_trains = 0

for station in manchester_line:
    current_api_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    api_url = f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepBoardWithDetails/{station}/{current_api_time}"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            live_data = response.json()
            trainServices = live_data.get('trainServices', [])
            
            station_delay_mins, station_delayed_trains = 0, 0
            print(f"\n📍 {station} DEPARTURES:")
            
            for train in trainServices[:6]: # Check the next 6 trains per station
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
                        # 🚀 THE FIX: Parse the full Staff API ISO-8601 Timestamp
                        try:
                            dt_std = datetime.fromisoformat(std)
                            dt_flag = datetime.fromisoformat(current_time_flag)
                        except ValueError:
                            # Fallback just in case Darwin ever reverts to HH:MM format
                            dt_std = datetime.strptime(str(std).strip()[:5], '%H:%M')
                            dt_flag = datetime.strptime(str(current_time_flag).strip()[:5], '%H:%M')
                        
                        delay = (dt_flag - dt_std).total_seconds() / 60
                        
                        if delay < -720: delay += 1440 
                        elif delay > 720: delay -= 1440
                        
                        if delay > 0:
                            station_delay_mins += delay
                            station_delayed_trains += 1
                            status = f"🔴 {int(delay)} mins late"
                            
                            if atd != 'N/A':
                                status += " (Departed)"
                        elif delay <= 0 and atd != 'N/A':
                            status = "🟢 On Time (Departed)"
                            
                    except Exception as e:
                        status = f"🟡 Unknown Format: [{current_time_flag}]"
                
                # Format the output to look clean in the console (strip out the date part)
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

# --- 2. DIRECTORY & AUTO-LOADER SETUP ---
base_path = r"C:\Users\OWNER\OneDrive - University of Keele\Disitation\code"
all_files = glob.glob(os.path.join(base_path, "data*.csv"))

if not all_files:
    print("❌ Error: No data files found. Check your folder path.")
    exit()

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# --- 3. DATA PREPARATION & ROUTE FILTERING ---
try:
    df = df[df['CRS'].isin(manchester_line)] 
    print(f"✅ Filtered historical data to route stations only (MAN, SPT, MAC, SOT, RUG, MKC, EUS).")
except KeyError:
    print("⚠️ WARNING: Could not find a 'CRS' column to filter by. Training on ALL data instead. Please check your CSV column names!")

df['DATE_STRING'] = df['EVENT_DATETIME'].astype(str).str[:10]
df['EVENT_DATETIME'] = pd.to_datetime(df['DATE_STRING'], errors='coerce', format='mixed', dayfirst=True)
df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')

daily_performance = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES', 'NON_PFPI_MINUTES']].sum().reset_index()
daily_performance['TOTAL_COMBINED_MINUTES'] = daily_performance['PFPI_MINUTES'] + daily_performance['NON_PFPI_MINUTES']
daily_performance['EVENT_DATETIME'] = pd.to_datetime(daily_performance['EVENT_DATETIME'])

total_days = len(daily_performance)

# --- 4. 80/20 SPLIT & SCALING ---
target_data = daily_performance.filter(['TOTAL_COMBINED_MINUTES'])
dataset = target_data.values
training_data_len = math.ceil(total_days * 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# --- 5. CREATING TRAINING DATA (80%) ---
window_size = 14 
train_data = scaled_data[0:training_data_len, :]

x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# --- 6. BUILDING & TRAINING THE AI ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2), 
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print(f"\n🧠 --- TRAINING AI ON MANCHESTER-EUSTON ROUTE (80% DATA) ---")
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0) 

# --- 7. TESTING THE AI (20% UNSEEN DATA) ---
print("🎯 --- EVALUATING TEST SET (20% DATA) ---")
test_data = scaled_data[training_data_len - window_size: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])

x_test = np.reshape(np.array(x_test), (len(x_test), window_size, 1))
test_predictions = scaler.inverse_transform(model.predict(x_test, verbose=0))

rmse = np.sqrt(np.mean(((test_predictions - y_test) ** 2)))
print(f"✅ Route RMSE on Test Data: {rmse:.2f} Minutes")

# --- 8. FUTURE FORECAST (NEXT 30 DAYS) ---
future_days = 30
future_predictions = []
current_window = scaled_data[-window_size:] 

for i in range(future_days):
    current_window_reshaped = np.reshape(current_window, (1, window_size, 1))
    next_day_scaled = model.predict(current_window_reshaped, verbose=0)
    future_predictions.append(next_day_scaled[0, 0])
    current_window = np.append(current_window[1:], next_day_scaled, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# --- 9. THE ZOOMED-IN ROUTE CHART ---
valid_plot = daily_performance[training_data_len:].copy()
valid_plot['Predictions'] = test_predictions

last_actual_date = daily_performance['EVENT_DATETIME'].iloc[-1]
future_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=future_days)

zoom_start_date = last_actual_date - pd.Timedelta(days=30)
zoomed_actuals = valid_plot[valid_plot['EVENT_DATETIME'] >= zoom_start_date]

plt.figure(figsize=(16, 7))
plt.title(f'Manchester to Euston Route: Latest 30 Days vs Upcoming 30-Day Forecast', fontsize=16, fontweight='bold')

plt.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['TOTAL_COMBINED_MINUTES'], 
         label='Actual Delay Minutes (Last 30 Days)', color='darkorange', linewidth=2.5, marker='o')

plt.plot(zoomed_actuals['EVENT_DATETIME'], zoomed_actuals['Predictions'], 
         label='AI Retrospective Test (Accuracy Check)', color='navy', linestyle='--', linewidth=2)

plt.plot(future_dates, future_predictions, 
         label='Future Route Forecast (Next 30 Days)', color='crimson', linestyle='-.', linewidth=2.5, marker='x')

plt.axvline(x=last_actual_date, color='black', linestyle='-', linewidth=2, label='Today (Data Cutoff)')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3)) 
plt.xticks(rotation=45)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Route Delay Minutes', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

chart_path = os.path.join(base_path, "Manchester_to_Euston_Forecast.png")
plt.savefig(chart_path, dpi=300)
print(f"✅ Route Chart auto-updated: {chart_path}")

# --- 10. DAILY ROUTE COEFFICIENT ---
historical_daily = daily_performance[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
historical_daily.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL_COMBINED_MINUTES': 'Minutes'}, inplace=True)
historical_daily['Data_Type'] = 'Actual'

future_daily = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()})
future_daily['Data_Type'] = 'Forecast'

master_daily = pd.concat([historical_daily, future_daily], ignore_index=True)
master_daily['Stress_Coefficient'] = (master_daily['Minutes'] / historical_daily['Minutes'].max()).clip(upper=1.0) 

csv_path = os.path.join(base_path, "coeficient.csv")
master_daily.to_csv(csv_path, index=False)
print(f"✅ Daily Coefficient Database auto-updated: {csv_path}")
print("\n🎉 ROUTE PIPELINE COMPLETE!")