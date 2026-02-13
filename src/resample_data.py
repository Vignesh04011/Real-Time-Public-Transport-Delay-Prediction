import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/gtfs_logs.csv")

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set index
df.set_index('Timestamp', inplace=True)

# Sort by time
df = df.sort_index()

# Resample to 5-minute intervals
df_resampled = df['Delay_Seconds'].resample('5min').mean()

# Handle missing values
df_resampled = df_resampled.interpolate()

# Save
df_resampled.to_csv("data/raw/resampled_delay.csv")

print("Resampling complete!")
print(df_resampled.head())
