import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/raw/resampled_delay.csv")

# Convert timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Use only delay values
data = df['Delay_Seconds'].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sliding window
X = []
y = []
window = 10

for i in range(len(scaled_data) - window):
    X.append(scaled_data[i:i+window])
    y.append(scaled_data[i+window])

X = np.array(X)
y = np.array(y)

# Save
np.save("data/raw/X.npy", X)
np.save("data/raw/y.npy", y)

print("LSTM data ready!")
print("Shape X:", X.shape)
print("Shape y:", y.shape)

import joblib
joblib.dump(scaler, "data/raw/scaler.save")
