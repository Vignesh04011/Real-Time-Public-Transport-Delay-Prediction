import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load resampled data
df = pd.read_csv("data/raw/resampled_delay.csv")

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df.set_index('Timestamp', inplace=True)

# Run ADF Test
result = adfuller(df['Delay_Seconds'])

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Data is STATIONARY ✅")
else:
    print("Data is NOT stationary ❌")
