import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load actual & predicted from LSTM
actual = np.load("data/raw/lstm_actual.npy")
predicted = np.load("data/raw/lstm_pred.npy")

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)
