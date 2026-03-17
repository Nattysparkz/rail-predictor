import pandas as pd
import os
import time
import sys
from datetime import datetime
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print(f"=======================================================")
print(f"📡 API HARVESTER: RUNNING CONTINUOUSLY")
print(f"=======================================================\n")

base_path = r"/mnt/c/Users/OWNER/OneDrive - University of Keele/Disitation/code" 
if not os.path.exists(base_path):
    base_path = r"C:\Users\OWNER\OneDrive - University of Keele\Disitation\code"

ongoing_csv_path = os.path.join(base_path, "data_ongoing_live.csv")

api_key = "EhPYIKPzBrWdoIqeA6u1hGc54eJSCcZxiGGgGqfGSwkwuGVQ"
manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS'] 
headers = {'x-apikey': api_key, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

print("✅ Harvester Online. Press [Ctrl + C] to safely stop it.")

try:
    while True: # INFINITE LOOP
        loop_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n🔄 [Scan at {loop_time}] Harvesting data across {len(manchester_line)} stations...")
        
        live_data_log = []
        
        for station in manchester_line:
            current_api_time = datetime.now().strftime("%Y%m%dT%H%M%S")
            api_url = f"https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120/GetDepBoardWithDetails/{station}/{current_api_time}?numRows=15&timeWindow=120"
            
            try:
                response = requests.get(api_url, headers=headers, timeout=10, verify=False)
                if response.status_code == 200:
                    live_data = response.json()
                    trainServices = live_data.get('trainServices', [])
                    station_delay_mins = 0
                    
                    for train in trainServices: 
                        std = train.get('std', 'N/A')
                        etd = train.get('etd', 'N/A')
                        atd = train.get('atd', 'N/A') 
                        is_cancelled = train.get('isCancelled', False)
                        
                        current_time_flag = atd if atd != 'N/A' else etd
                        flag_lower = str(current_time_flag).lower().strip()
                        
                        if is_cancelled or flag_lower == 'cancelled':
                            station_delay_mins += 60 
                        elif flag_lower not in ['on time', 'n/a', 'no report', 'delayed'] and std != 'N/A':
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
                            except Exception:
                                pass
                    
                    live_data_log.append({
                        'EVENT_DATETIME': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'CRS': station,
                        'PFPI_MINUTES': station_delay_mins,
                        'NON_PFPI_MINUTES': 0
                    })
            except Exception:
                pass 
                
        # Save to CSV immediately
        if live_data_log:
            ongoing_df = pd.DataFrame(live_data_log)
            try:
                if os.path.exists(ongoing_csv_path):
                    ongoing_df.to_csv(ongoing_csv_path, mode='a', header=False, index=False)
                else:
                    ongoing_df.to_csv(ongoing_csv_path, mode='w', header=True, index=False)
                print(f"   ↳ 💾 Saved {len(live_data_log)} rows to data_ongoing_live.csv")
            except PermissionError:
                print("   ⚠️ ERROR: Could not save data! Please close 'data_ongoing_live.csv' if it is open in Excel.")

        # Sleep for 60 seconds before next scan
        for remaining in range(60, 0, -1):
            sys.stdout.write(f"\r⏳ Next scan in {remaining:02d} seconds...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r" + " " * 50 + "\r") 

except KeyboardInterrupt:
    print("\n\n🛑 Scraper manually stopped. Data is safe!")