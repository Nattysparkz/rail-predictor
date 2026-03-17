import os
import sys

# ---  VENV BOOTSTRAP ---
# If not running inside the project venv, re-launch using it so that all
# dependencies (confluent-kafka, jax, flax, etc.) are available.
_SCRIPT  = os.path.abspath(__file__)
_VENV_PY = os.path.join(os.path.dirname(_SCRIPT), 'ai_env', 'bin', 'python3')
_VENV_DIR = os.path.join(os.path.dirname(_SCRIPT), 'ai_env')
if os.path.isfile(_VENV_PY) and not sys.prefix.startswith(_VENV_DIR):
    print(f"⚙️  Relaunching with venv Python: {_VENV_PY}")
    import subprocess
    raise SystemExit(subprocess.call([_VENV_PY] + sys.argv))

# ---  SYSTEM CHECK & SILENCE WARNINGS ---
# THIS MUST BE AT THE VERY TOP BEFORE ANY JAX IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'

import pandas as pd
import numpy as np
import glob
import uuid
import math
import time
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import urllib3
from sklearn.preprocessing import MinMaxScaler
from confluent_kafka import Consumer, KafkaError, TopicPartition, OFFSET_BEGINNING

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
print(f"📡 DATA SOURCE: DARWIN REAL-TIME PUSH PORT (KAFKA)")
print(f"🕒 System Time: {current_time_display}")
print(f"=======================================================\n")

# ---  ROUTE DEFINITION ---
manchester_line = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']

# Darwin uses TIPLOC codes; map CRS → TIPLOC for TS message filtering
CRS_TO_TIPLOC = {
    'MAN': 'MNCRPIC',
    'SPT': 'STKP',
    'MAC': 'MACLSFD',
    'SOT': 'STOKEOT',
    'CRE': 'CREWE',
    'RUG': 'RUGBY',
    'MKC': 'MLTKNCS',
    'EUS': 'EUSTON',
}
TIPLOC_TO_CRS = {v: k for k, v in CRS_TO_TIPLOC.items()}
ROUTE_TIPLOCS = set(CRS_TO_TIPLOC.values())

# ---  DARWIN KAFKA CONFIGURATION ---
KAFKA_BOOTSTRAP  = 'pkc-z3p1v0.europe-west2.gcp.confluent.cloud:9092'
KAFKA_USERNAME   = 'GZSSPQNWCOTEVH4R'
KAFKA_PASSWORD   = 'cfltITE1NBLRmcGJsqzwXQpkRoYLreyhf2fZwqxNZXvAInxsEy1T2rd+S0+GEDaQ'
KAFKA_TOPIC_JSON = 'prod-1010-Darwin-Train-Information-Push-Port-IIII2_0-JSON'
KAFKA_GROUP      = 'SC-de021cfb-7f0b-4996-8f27-bf8c68793fcd'
CONSUME_SECONDS  = 120  # listen for 2 minutes

base_path  = r"/mnt/c/Users/OWNER/OneDrive - University of Keele/Disitation/code"
DARWIN_CSV = os.path.join(base_path, 'darwin_live.csv')


# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def parse_hhmm_to_td(t_str):
    """Convert HH:MM, HH:MM:SS, or HHMM strings to timedelta. Returns None on failure."""
    if not t_str:
        return None
    try:
        s = str(t_str).strip()
        parts = s.split(':')
        if len(parts) == 1:
            padded = parts[0].zfill(4)
            h, m = int(padded[:2]), int(padded[2:4])
            sec = int(padded[4:6]) if len(padded) >= 6 else 0
        else:
            h, m = int(parts[0]), int(parts[1])
            sec = int(parts[2]) if len(parts) > 2 else 0
        return timedelta(hours=h, minutes=m, seconds=sec)
    except Exception:
        return None


def calc_delay_minutes(scheduled_str, reported_str):
    sched = parse_hhmm_to_td(scheduled_str)
    rep   = parse_hhmm_to_td(reported_str)
    if sched is None or rep is None:
        return 0.0
    diff = (rep - sched).total_seconds() / 60.0
    if diff < -720: diff += 1440
    elif diff > 720: diff -= 1440
    return diff


def get_attr(d, key):
    return d.get(key) or d.get(f'@{key}')


def extract_ts_elements(obj):
    found = []
    if isinstance(obj, dict):
        if 'TS' in obj:
            ts_val = obj['TS']
            found.extend(ts_val if isinstance(ts_val, list) else [ts_val])
        for v in obj.values():
            found.extend(extract_ts_elements(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(extract_ts_elements(item))
    return found


# -----------------------------------------------------------------------
# STAGE 1 — DARWIN KAFKA PUSH PORT
# -----------------------------------------------------------------------

def consume_darwin_delays(seconds=CONSUME_SECONDS):
    # Fresh group ID every run so OFFSET_BEGINNING always takes effect
    group = f"{KAFKA_GROUP}-{uuid.uuid4().hex[:8]}"

    kafka_conf = {
        'bootstrap.servers':  KAFKA_BOOTSTRAP,
        'sasl.mechanisms':    'PLAIN',
        'security.protocol':  'SASL_SSL',
        'sasl.username':      KAFKA_USERNAME,
        'sasl.password':      KAFKA_PASSWORD,
        'group.id':           group,
        'auto.offset.reset':  'earliest',
        'enable.auto.commit': False,
        'session.timeout.ms': 45000,
    }

    def _on_assign(consumer, partitions):
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)
        print(f"  ✅ Assigned {len(partitions)} partition(s), seeking to beginning…")

    consumer = Consumer(kafka_conf)
    consumer.subscribe([KAFKA_TOPIC_JSON], on_assign=_on_assign)

    station_summary = {crs: {'delay_mins': 0.0, 'delayed': 0} for crs in manchester_line}
    total_delay_mins   = 0.0
    total_delayed      = 0
    msgs_received      = 0
    tiploc_hits        = 0
    no_sched_count     = 0
    no_rep_count       = 0
    _sample_loc        = None
    _raw_sample        = None

    print(f"📡 --- SCANNING DARWIN LIVE FEED ({seconds}s window) ---")
    deadline  = time.time() + seconds
    next_tick = time.time() + 10

    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = consumer.poll(timeout=min(remaining, 1.0))

            if time.time() >= next_tick:
                elapsed = int(seconds - remaining)
                print(f"  ⏳ {elapsed}s elapsed … {msgs_received} msgs so far", flush=True)
                next_tick += 10

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"⚠️ Kafka error: {msg.error()}")
                    break
                continue

            msgs_received += 1

            try:
                payload = json.loads(msg.value().decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if _raw_sample is None:
                _raw_sample = payload

            for ts in extract_ts_elements(payload):
                locations = ts.get('Location') or ts.get('@Location', [])
                if isinstance(locations, dict):
                    locations = [locations]
                if not isinstance(locations, list):
                    continue

                for loc in locations:
                    tpl = str(get_attr(loc, 'tpl') or '').strip()
                    if tpl not in ROUTE_TIPLOCS:
                        continue
                    crs = TIPLOC_TO_CRS.get(tpl)
                    if not crs:
                        continue

                    tiploc_hits += 1
                    if _sample_loc is None:
                        _sample_loc = loc

                    sched = (get_attr(loc, 'ptd') or get_attr(loc, 'wtd') or
                             get_attr(loc, 'pta') or get_attr(loc, 'wta'))
                    if not sched:
                        no_sched_count += 1
                        continue

                    dep_node = loc.get('dep') or loc.get('arr') or loc.get('pass') or {}
                    reported = None
                    if isinstance(dep_node, dict):
                        reported = (get_attr(dep_node, 'at') or
                                    get_attr(dep_node, 'et') or
                                    get_attr(dep_node, 'wet'))
                    if not reported:
                        no_rep_count += 1
                        continue

                    delay = calc_delay_minutes(sched, reported)
                    if delay > 0:
                        station_summary[crs]['delay_mins'] += delay
                        station_summary[crs]['delayed']    += 1
                        total_delay_mins += delay
                        total_delayed    += 1
    finally:
        consumer.close()

    # --- Per-station summary ---
    print()
    live_rows = []
    for crs in manchester_line:
        s = station_summary[crs]
        if s['delayed'] > 0:
            print(f"  📍 {crs}: {s['delayed']} delayed, ~{s['delay_mins']:.0f} mins")
        else:
            print(f"  📍 {crs}: 🟢 No delays detected")
        live_rows.append({
            'EVENT_DATETIME':   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'CRS':              crs,
            'PFPI_MINUTES':     s['delay_mins'],
            'NON_PFPI_MINUTES': 0,
        })

    # --- Save to darwin_live.csv ---
    df_live = pd.DataFrame(live_rows)
    if os.path.exists(DARWIN_CSV):
        df_live.to_csv(DARWIN_CSV, mode='a', header=False, index=False)
    else:
        df_live.to_csv(DARWIN_CSV, mode='w', header=True, index=False)
    print(f"  💾 Saved {len(live_rows)} rows → darwin_live.csv")

    print("-" * 55)
    print(f"📬 Messages processed : {msgs_received}")
    print(f"📍 TIPLOC hits        : {tiploc_hits}")
    print(f"⚠️  No sched time     : {no_sched_count}")
    print(f"⚠️  No reported time  : {no_rep_count}")
    print(f"📊 DARWIN SNAPSHOT: {total_delayed} delayed, ~{total_delay_mins:.0f} mins\n")

    if _raw_sample is not None and _sample_loc is None:
        print("🔍 DEBUG — first raw message (no route TIPLOC match):")
        print(json.dumps(_raw_sample, indent=2, default=str)[:1500])
        print()
    if _sample_loc is not None:
        print("🔍 DEBUG — first matching route location:")
        print(json.dumps(_sample_loc, indent=2, default=str))
        print()

    return total_delay_mins, total_delayed


SCAN_INTERVAL = 300  # seconds to wait between pipeline runs

_run_count = 0
print("🔁 Running continuously — press Ctrl+C to stop.\n")

while True:
    try:
        _run_count += 1
        print(f"\n{'='*55}")
        print(f"🔄 RUN #{_run_count}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*55}\n")

        # -----------------------------------------------------------------------
        # STAGE 1 — DARWIN KAFKA PUSH PORT
        # -----------------------------------------------------------------------

        total_line_delay_mins, total_line_delayed_trains = consume_darwin_delays()

        # -----------------------------------------------------------------------
        # STAGE 2 — LOAD HISTORICAL CSV DATA
        # -----------------------------------------------------------------------

        all_files = glob.glob(os.path.join(base_path, "data*.csv"))
        if not all_files:
            _alt = r"C:\Users\OWNER\OneDrive - University of Keele\Disitation\code"
            all_files = glob.glob(os.path.join(_alt, "data*.csv"))
        if not all_files:
            print("❌ Error: No data files found. Retrying next scan.")
            time.sleep(SCAN_INTERVAL)
            continue

        print(f"📂 Found {len(all_files)} files. Checking for corrupted/empty files...")
        df_list = []
        for f in all_files:
            try:
                if os.path.getsize(f) > 0:
                    df_list.append(pd.read_csv(f, low_memory=False))
            except Exception:
                pass

        if not df_list:
            print("❌ CRITICAL ERROR: All CSV files empty or corrupted. Retrying next scan.")
            time.sleep(SCAN_INTERVAL)
            continue

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
            print(f"✅ Filtered to route stations using column: '{station_col}'.")
        else:
            print("⚠️ WARNING: No station column found. Training on ALL data.")

        df['DATE_STRING']    = df['EVENT_DATETIME'].astype(str).str[:10]
        df['EVENT_DATETIME'] = pd.to_datetime(df['DATE_STRING'], errors='coerce',
                                              format='mixed', dayfirst=True)
        df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')

        if 'NON_PFPI_MINUTES' not in df.columns:
            df['NON_PFPI_MINUTES'] = 0
        if 'PFPI_MINUTES' not in df.columns:
            df['PFPI_MINUTES'] = 0

        daily_performance = (df.groupby(df['EVENT_DATETIME'].dt.date)
                               [['PFPI_MINUTES', 'NON_PFPI_MINUTES']]
                               .sum()
                               .reset_index())
        daily_performance['TOTAL_COMBINED_MINUTES'] = (daily_performance['PFPI_MINUTES'] +
                                                       daily_performance['NON_PFPI_MINUTES'])
        daily_performance['EVENT_DATETIME'] = pd.to_datetime(daily_performance['EVENT_DATETIME'])

        # -----------------------------------------------------------------------
        # STAGE 3 — INJECT LIVE DARWIN DATA INTO HISTORICAL SERIES
        # -----------------------------------------------------------------------

        today_date_pd = pd.to_datetime(datetime.now().date())

        if today_date_pd in daily_performance['EVENT_DATETIME'].values:
            idx = daily_performance.index[daily_performance['EVENT_DATETIME'] == today_date_pd][0]
            daily_performance.at[idx, 'TOTAL_COMBINED_MINUTES'] += total_line_delay_mins
            print(f"📌 Updated today's row with {total_line_delay_mins:.0f} Darwin delay minutes.")
        else:
            new_row = pd.DataFrame({
                'EVENT_DATETIME':        [today_date_pd],
                'PFPI_MINUTES':          [total_line_delay_mins],
                'NON_PFPI_MINUTES':      [0],
                'TOTAL_COMBINED_MINUTES':[total_line_delay_mins],
            })
            daily_performance = pd.concat([daily_performance, new_row], ignore_index=True)
            daily_performance = (daily_performance
                                 .sort_values('EVENT_DATETIME')
                                 .reset_index(drop=True))
            print(f"📌 Appended today's Darwin snapshot as a new row.")

        total_days = len(daily_performance)

        # -----------------------------------------------------------------------
        # STAGE 4 — DATA PREPARATION & WINDOWING
        # -----------------------------------------------------------------------

        target_data       = daily_performance.filter(['TOTAL_COMBINED_MINUTES'])
        dataset           = target_data.values
        training_data_len = math.ceil(total_days * 0.8)

        scaler      = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        window_size = 14
        train_data  = scaled_data[0:training_data_len, :]

        x_train, y_train = [], []
        for i in range(window_size, len(train_data)):
            x_train.append(train_data[i - window_size:i, 0])
            y_train.append(train_data[i, 0])

        x_train_jnp = jnp.array(x_train).reshape(-1, window_size, 1)
        y_train_jnp = jnp.array(y_train).reshape(-1, 1)

        # -----------------------------------------------------------------------
        # STAGE 5 — JAX/FLAX NEURAL NETWORK DEFINITION
        # -----------------------------------------------------------------------

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

        model     = JaxRouteModel()
        key       = random.PRNGKey(42)
        variables = model.init(key, jnp.ones((1, window_size, 1)))
        params    = variables['params']
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(params)

        # -----------------------------------------------------------------------
        # STAGE 6 — JIT-COMPILED TRAINING ENGINE
        # -----------------------------------------------------------------------

        @jax.jit
        def train_step(params, opt_state, x, y):
            def loss_fn(p):
                preds = model.apply({'params': p}, x)
                return jnp.mean((preds - y) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        print(f"\n🧠 --- TRAINING JAX AI ON MANCHESTER-EUSTON ROUTE ---")
        for epoch in range(51):
            start_t = time.time()
            params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
            if epoch % 10 == 0:
                loss.block_until_ready()
                dur = (time.time() - start_t) * 1000
                print(f"Epoch {epoch:02d} | Loss: {loss:.6f} | Compute Speed: {dur:.2f}ms")

        # -----------------------------------------------------------------------
        # STAGE 7 — EVALUATION ON 20% HELD-OUT TEST SET
        # -----------------------------------------------------------------------

        print("\n🎯 --- EVALUATING JAX TEST SET (20% DATA) ---")
        test_data   = scaled_data[training_data_len - window_size:, :]
        x_test      = [test_data[i - window_size:i, 0]
                       for i in range(window_size, len(test_data))]
        x_test_jnp  = jnp.array(x_test).reshape(-1, window_size, 1)

        y_test           = dataset[training_data_len:, :]
        preds_scaled     = model.apply({'params': params}, x_test_jnp)
        test_predictions = scaler.inverse_transform(np.array(preds_scaled))

        rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
        print(f"  Route RMSE on Test Data: {rmse:.2f} Minutes")

        # -----------------------------------------------------------------------
        # STAGE 8 — 30-DAY AUTOREGRESSIVE FORECAST
        # -----------------------------------------------------------------------

        future_days        = 30
        future_predictions = []
        current_window     = scaled_data[-window_size:]

        for i in range(future_days):
            in_win    = jnp.reshape(current_window, (1, window_size, 1))
            next_pred = model.apply({'params': params}, in_win)
            future_predictions.append(next_pred[0, 0])
            current_window = jnp.append(current_window[1:], next_pred, axis=0)

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        # -----------------------------------------------------------------------
        # STAGE 9 — CHART GENERATION
        # -----------------------------------------------------------------------

        valid_plot              = daily_performance[training_data_len:].copy()
        valid_plot['Predictions'] = test_predictions

        last_actual_date = daily_performance['EVENT_DATETIME'].iloc[-1]
        future_dates     = pd.date_range(
            start=last_actual_date + pd.Timedelta(days=1), periods=future_days
        )
        zoom_start_date  = last_actual_date - pd.Timedelta(days=30)
        zoomed_actuals   = valid_plot[valid_plot['EVENT_DATETIME'] >= zoom_start_date]

        plt.figure(figsize=(16, 7))
        plt.title('Manchester to Euston: JAX Model 30-Day Forecast (Darwin Push Port)',
                  fontsize=16, fontweight='bold')
        plt.plot(zoomed_actuals['EVENT_DATETIME'],
                 zoomed_actuals['TOTAL_COMBINED_MINUTES'],
                 label='Actual Delay Minutes (Last 30 Days)',
                 color='darkorange', linewidth=2.5, marker='o')
        plt.plot(zoomed_actuals['EVENT_DATETIME'],
                 zoomed_actuals['Predictions'],
                 label='JAX Retrospective Test (Accuracy Check)',
                 color='navy', linestyle='--', linewidth=2)
        plt.plot(future_dates, future_predictions,
                 label='Future JAX Forecast (Next 30 Days)',
                 color='crimson', linestyle='-.', linewidth=2.5, marker='x')
        plt.axvline(x=last_actual_date, color='black', linestyle='-',
                    linewidth=2, label='Today (Darwin Data Cutoff)')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.xticks(rotation=45)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Route Delay Minutes', fontsize=12)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

        chart_path = os.path.join(base_path, "Manchester_to_Euston_Darwin_JAX_Forecast.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"\n  Route chart saved: {chart_path}")

        # -----------------------------------------------------------------------
        # STAGE 10 — DAILY STRESS COEFFICIENT EXPORT
        # -----------------------------------------------------------------------

        historical_daily = daily_performance[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
        historical_daily.rename(columns={
            'EVENT_DATETIME':        'Date',
            'TOTAL_COMBINED_MINUTES':'Minutes'
        }, inplace=True)
        historical_daily['Data_Type'] = 'Actual'

        future_daily = pd.DataFrame({
            'Date':      future_dates,
            'Minutes':   future_predictions.flatten(),
            'Data_Type': 'Forecast',
        })

        master_daily = pd.concat([historical_daily, future_daily], ignore_index=True)
        master_daily['Stress_Coefficient'] = (
            master_daily['Minutes'] / historical_daily['Minutes'].max()
        ).clip(upper=1.0)

        csv_path = os.path.join(base_path, "jax_darwin_coeficient.csv")
        master_daily.to_csv(csv_path, index=False)
        print(f"  Stress coefficient CSV saved: {csv_path}")

        print(f"\n🎉 RUN #{_run_count} COMPLETE — sleeping {SCAN_INTERVAL}s until next scan…")
        time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n👋 Pipeline stopped. Goodbye!")
        break
