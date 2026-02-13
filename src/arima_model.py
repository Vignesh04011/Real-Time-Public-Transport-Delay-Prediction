import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/raw/resampled_delay.csv")

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Train-Test Split
train = df.iloc[:int(len(df)*0.8)]
test = df.iloc[int(len(df)*0.8):]

# Build ARIMA Model
model = ARIMA(train['Delay_Seconds'], order=(2,0,2))
model_fit = model.fit()

# Predict
predictions = model_fit.forecast(steps=len(test))

# Plot
plt.figure(figsize=(10,5))
plt.plot(test.index, test['Delay_Seconds'], label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("ARIMA Prediction")
plt.show()
