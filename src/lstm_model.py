import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import joblib
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
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

# ✅ SAVE MODEL AFTER TRAINING
model.save("outputs/lstm_model.keras")

# Predict
pred = model.predict(X_test)

pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

np.save("data/raw/lstm_pred.npy", pred)
np.save("data/raw/lstm_actual.npy", y_test)

# ✅ PREDICT FUNCTION (used by Streamlit)
def predict_lstm(input_sequence):
    model = load_model("outputs/lstm_model.keras", compile=False)
    prediction = model.predict(input_sequence)
    return float(prediction[0][0])

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("LSTM Prediction")
plt.show()