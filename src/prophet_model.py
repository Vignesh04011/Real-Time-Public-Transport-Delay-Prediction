import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/raw/resampled_delay.csv")

# Prophet format
df.rename(columns={'Timestamp':'ds','Delay_Seconds':'y'}, inplace=True)

# Train-Test split
train = df.iloc[:int(len(df)*0.8)]
test = df.iloc[int(len(df)*0.8):]

# Model
model = Prophet()
model.fit(train)

# Future prediction
future = model.make_future_dataframe(periods=len(test), freq='5min')
forecast = model.predict(future)

# Convert ds to datetime for safe plotting
test['ds'] = pd.to_datetime(test['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])

plt.figure(figsize=(10,5))
plt.plot(test['ds'], test['y'], label="Actual")
plt.plot(forecast['ds'].iloc[-len(test):], forecast['yhat'].iloc[-len(test):], label="Predicted")
plt.legend()
plt.title("Prophet Prediction")
plt.show()

