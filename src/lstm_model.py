import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data
X = np.load("data/raw/X.npy")
y = np.load("data/raw/y.npy")

scaler = joblib.load("data/raw/scaler.save")

# Train-test split
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(10,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=20, batch_size=16)

# Predict
pred = model.predict(X_test)

pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)

np.save("data/raw/lstm_pred.npy", pred)
np.save("data/raw/lstm_actual.npy", y_test)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("LSTM Prediction")
plt.show()
