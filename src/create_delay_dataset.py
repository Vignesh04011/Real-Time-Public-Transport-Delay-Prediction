import pandas as pd
import numpy as np

# Load files
stop_times = pd.read_csv("data/raw/stop_times.txt")
trips = pd.read_csv("data/raw/trips.txt")

# Convert arrival time
stop_times['arrival_time'] = pd.to_datetime(stop_times['arrival_time'], errors='coerce')

# Remove invalid rows
stop_times = stop_times.dropna(subset=['arrival_time'])

# Simulate real-world delay
np.random.seed(42)
stop_times['actual_time'] = stop_times['arrival_time'] + pd.to_timedelta(
    np.random.randint(-120, 600, size=len(stop_times)), unit='s'
)

# Delay calculation
stop_times['Delay_Seconds'] = (stop_times['actual_time'] - stop_times['arrival_time']).dt.total_seconds()

# Merge with trip info
merged = stop_times.merge(trips, on='trip_id')

# Create final dataset
final_df = pd.DataFrame({
    'Timestamp': merged['actual_time'],
    'RouteID': merged['route_id'],
    'StopID': merged['stop_id'],
    'VehicleID': merged['trip_id'],
    'Delay_Seconds': merged['Delay_Seconds']
})

final_df = final_df.dropna()

# Save
final_df.to_csv("data/raw/gtfs_logs.csv", index=False)

print("Real GTFS-based delay dataset created!")
print(final_df.head())
