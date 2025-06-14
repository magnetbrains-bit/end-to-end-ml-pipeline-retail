# scripts/process_data.py (V5 - Final with inf cleanup)

import pandas as pd
from datetime import timedelta
import os
import numpy as np # Import numpy for inf

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE_PATH = os.path.join(DATA_DIR, "events.csv")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "training_data.parquet")

# --- Feature and Target Functions ---
def create_features_v2(events, end_date, time_windows=[1, 7, 30]):
    df = events[events['timestamp'] <= end_date].copy()
    visitor_features = pd.DataFrame(df['visitorid'].unique(), columns=['visitorid'])
    for days in time_windows:
        start_date_window = end_date - timedelta(days=days)
        window_df = df[df['timestamp'] >= start_date_window]
        agg_features = window_df.groupby('visitorid').agg(
            total_events=('event', 'count'), num_views=('event', lambda x: (x == 'view').sum()),
            num_addtocart=('event', lambda x: (x == 'addtocart').sum()), num_unique_items=('itemid', 'nunique')
        ).reset_index()
        agg_features.columns = ['visitorid', f'total_events_{days}d', f'num_views_{days}d', f'num_addtocart_{days}d', f'num_unique_items_{days}d']
        visitor_features = pd.merge(visitor_features, agg_features, on='visitorid', how='left')
    recency_df = df.groupby('visitorid')['timestamp'].max().reset_index()
    recency_df.columns = ['visitorid', 'last_event_ts']
    visitor_features = pd.merge(visitor_features, recency_df, on='visitorid', how='left')
    visitor_features['days_since_last_event'] = (end_date - visitor_features['last_event_ts']).dt.days
    visitor_features.drop(columns=['last_event_ts'], inplace=True)
    visitor_features.fillna(0, inplace=True)
    
    visitor_features['add_to_cart_rate_7d'] = visitor_features['num_addtocart_7d'] / visitor_features['num_views_7d']
    
    # --- THIS IS THE FIX ---
    # Replace any 'inf' values that result from division by zero with 0.
    visitor_features.replace([np.inf, -np.inf], 0, inplace=True)
    visitor_features.fillna(0, inplace=True) # Also run fillna again just in case
    
    return visitor_features

def create_target(events, users, start_date, end_date):
    target_window = events[(events['timestamp'] > start_date) & (events['timestamp'] <= end_date)]
    buyers = target_window[target_window['event'] == 'transaction']['visitorid'].unique()
    users['target'] = users['visitorid'].isin(buyers).astype(int)
    return users

def main():
    print("Starting data processing...")
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Input data file not found at {INPUT_FILE_PATH}")
        exit(1)
    dtype_spec = {'visitorid': 'int32', 'itemid': 'int32', 'transactionid': 'float32'}
    events_df = pd.read_csv(INPUT_FILE_PATH, dtype=dtype_spec)
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
    print("Data loaded.")
    train_end_date = pd.to_datetime('2015-08-15')
    target_start_date = train_end_date
    target_end_date = train_end_date + timedelta(days=7)
    print("Creating features...")
    features_df = create_features_v2(events_df, train_end_date)
    print("Creating target labels...")
    training_data = create_target(events_df, features_df, target_start_date, target_end_date)
    os.makedirs(DATA_DIR, exist_ok=True)
    training_data.to_parquet(OUTPUT_FILE_PATH, index=False)
    print(f"Training data saved to {OUTPUT_FILE_PATH}")
    print("Data processing complete.")

if __name__ == "__main__":
    main()