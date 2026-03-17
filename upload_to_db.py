"""
Upload CSV files to Digital Ocean PostgreSQL Database
=====================================================
Usage:
    pip install psycopg2-binary pandas
    python upload_to_db.py --db-url "postgresql://user:pass@host:25060/defaultdb?sslmode=require"
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
                source_file TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_event_datetime ON rail_events(event_datetime);
        """)
    conn.commit()
    print("✅ Table 'rail_events' ready.")

def upload_csv(conn, filepath):
    filename = filepath.split("\\")[-1].split("/")[-1]
    print(f"📄 Loading {filename}...")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    if 'EVENT_DATETIME' not in df.columns or 'PFPI_MINUTES' not in df.columns:
        print(f"   ⚠️  Skipping {filename} — missing EVENT_DATETIME or PFPI_MINUTES columns")
        return 0
    
    # Parse dates - handles "05-JAN-2025 10:59" format
    df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], format='mixed', dayfirst=True, errors='coerce')
    df['PFPI_MINUTES'] = pd.to_numeric(df['PFPI_MINUTES'], errors='coerce')
    df = df.dropna(subset=['EVENT_DATETIME', 'PFPI_MINUTES'])
    
    if df.empty:
        print(f"   ⚠️  Skipping {filename} — no valid rows after parsing")
        return 0
    
    rows = [(row['EVENT_DATETIME'], row['PFPI_MINUTES'], filename) for _, row in df.iterrows()]
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO rail_events (event_datetime, pfpi_minutes, source_file) VALUES %s",
            rows,
            page_size=1000
        )
    conn.commit()
    print(f"   ✅ Uploaded {len(rows):,} rows from {filename}")
    return len(rows)

def main():
    parser = argparse.ArgumentParser(description="Upload CSVs to Digital Ocean PostgreSQL")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection string")
    parser.add_argument("--csv-dir", default=".", help="Directory containing CSV files (default: current dir)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before uploading")
    args = parser.parse_args()
    
    print("🔌 Connecting to database...")
    conn = psycopg2.connect(args.db_url)
    
    if args.clear:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS rail_events;")
        conn.commit()
        print("🗑️  Cleared existing data.")
    
    create_table(conn)
    
    csv_files = glob.glob(f"{args.csv_dir}/*.csv")
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
