import streamlit as st
import numpy as np
import pandas as pd
import joblib

from src.lstm_model import predict_lstm
from src.arima_model import predict_arima
from src.sarima_model import predict_sarima

st.set_page_config(page_title="Transport Delay Predictor")

st.title("🚍 Real-Time Transport Delay Prediction")

st.sidebar.header("Input Parameters")

route = st.sidebar.selectbox("Route ID", [101,102,103])
stop = st.sidebar.selectbox("Stop ID", [1,2,3,4])
time_index = st.sidebar.slider("Time Window", 1, 20, 10)

if st.sidebar.button("Predict Delay"):

    # 🔹 Load latest real delay values
    df = pd.read_csv("data/raw/resampled_delay.csv")

    last_10 = df['Delay_Seconds'].values[-10:].reshape(-1,1)

    # 🔹 Load scaler and scale input
    scaler = joblib.load("data/raw/scaler.save")
    scaled = scaler.transform(last_10)

    input_sequence = scaled.reshape(1,10,1)

    # 🔹 Predictions
    lstm_delay = predict_lstm(input_sequence)
    arima_delay = predict_arima()
    sarima_delay = predict_sarima()

    st.subheader("Predicted Delay")

    col1, col2, col3 = st.columns(3)

    col1.metric("LSTM", f"{round(lstm_delay,2)} sec")
    col2.metric("ARIMA", f"{round(arima_delay,2)} sec")
    col3.metric("SARIMA", f"{round(sarima_delay,2)} sec")