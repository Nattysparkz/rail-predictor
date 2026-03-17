"""
Upload CSV files to Digital Ocean PostgreSQL Database
"""

import argparse
import glob
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rail_events (
                id SERIAL PRIMARY KEY,
                event_datetime TIMESTAMP,
                pfpi_minutes DOUBLE PRECISION,
                non_pfpi_minutes DOUBLE PRECISION,
                source_file TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_event_datetime ON rail_events(event_datetime);
        """)
    conn.commit()
    print("✅ Table 'rail_events' ready.")

def upload_csv(conn, filepath):
    filename = filepath.split("\\")[-1].split("/")[-1]
    print(f"📄 Loading {filename}...")
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"   ⚠️  Could not read {filename}: {e}")
        return 0
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.strip('"')
    
    if 'EVENT_DATETIME' not in df.columns:
        print(f"   ⚠️  Skipping {filename} — no EVENT_DATETIME column")
        print(f"   Columns found: {list(df.columns[:10])}")
        return 0
    
    if 'PFPI_MINUTES' not in df.columns:
        print(f"   ⚠️  Skipping {filename} — no PFPI_MINUTES column")
        return 0
    
    # Strip quotes from values
    df['EVENT_DATETIME'] = df['EVENT_DATETIME'].astype(str).str.strip().str.strip('"')
    
    # Debug
    print(f"   Sample dates: {df['EVENT_DATETIME'].head(3).tolist()}")
    print(f"   Rows before parse: {len(df)}")
    
    # Parse dates - handles "05-JAN-2025 10:59" format
    df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], format='mixed', dayfirst=True, errors='coerce')
    df['PFPI_MINUTES'] = pd.to_numeric(df['PFPI_MINUTES'], errors='coerce').fillna(0)
    
    if 'NON_PFPI_MINUTES' in df.columns:
        df['NON_PFPI_MINUTES'] = pd.to_numeric(df['NON_PFPI_MINUTES'], errors='coerce').fillna(0)
    else:
        df['NON_PFPI_MINUTES'] = 0
    
    valid = df.dropna(subset=['EVENT_DATETIME'])
    print(f"   Valid rows after parse: {len(valid)}")
    
    if valid.empty:
        print(f"   ⚠️  Skipping {filename} — no valid rows")
        return 0
    
    rows = [
        (row['EVENT_DATETIME'], row['PFPI_MINUTES'], row['NON_PFPI_MINUTES'], filename)
        for _, row in valid.iterrows()
    ]
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO rail_events (event_datetime, pfpi_minutes, non_pfpi_minutes, source_file) VALUES %s",
            rows,
            page_size=1000
        )
    conn.commit()
    print(f"   ✅ Uploaded {len(rows):,} rows from {filename}")
    return len(rows)

def main():
    parser = argparse.ArgumentParser(description="Upload CSVs to Digital Ocean PostgreSQL")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection string")
    parser.add_argument("--csv-dir", default=".", help="Directory containing CSV files")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before uploading")
    args = parser.parse_args()
    
    print("🔌 Connecting to database...")
    conn = psycopg2.connect(args.db_url)
    print("✅ Connected!\n")
    
    if args.clear:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS rail_events;")
        conn.commit()
        print("🗑️  Cleared existing data.\n")
    
    create_table(conn)
    
    csv_files = sorted(glob.glob(f"{args.csv_dir}/*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {args.csv_dir}")
        sys.exit(1)
    
    print(f"📂 Found {len(csv_files)} CSV files\n")
    
    total = 0
    for f in csv_files:
        total += upload_csv(conn, f)
    
    print(f"\n🎉 Done! Uploaded {total:,} total rows to rail_events table.")
    conn.close()

if __name__ == "__main__":
    main()
