import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Bidirectional
import yfinance as yf

# Define the stock symbol
stock_symbol = 'AAPL'

# Fetch more historical stock data for better training
start = dt.datetime(2023, 1, 1)  # Start earlier to get more training data
end = dt.datetime(2024, 8, 31)   # Including August for predictions

# Fetching historical stock data
data = yf.download(stock_symbol, start=start, end=end)

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Creating the training data (up to July 2024)
train_data_end = dt.datetime(2024, 7, 31)
train_data_index = np.where(data.index <= train_data_end)[0]
last_train_index = train_data_index[-1]

train_data = scaled_data[:last_train_index + 1]
x_train, y_train = [], []
prediction_days = 120   
for x in range(prediction_days, len(train_data)):
    x_train.append(train_data[x-prediction_days:x, 0])
    y_train.append(train_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Define the models
def build_model(model_type):
    model = Sequential()
    if model_type == 'simple_rnn':
        model.add(SimpleRNN(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(SimpleRNN(units=100))
        model.add(Dropout(0.3))
    elif model_type == 'lstm':
        model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100))
        model.add(Dropout(0.3))
    elif model_type == 'bilstm':
        model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(units=100)))
        model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train models with more epochs and adjusted batch size
models = ['simple_rnn', 'lstm', 'bilstm']
predictions = {}
for m in models:
    model = build_model(m)
    model.fit(x_train, y_train, epochs=100, batch_size=16)  # Increase epochs and adjust batch size
    
    # Prepare test data for prediction
    test_data = scaled_data[last_train_index + 1 - prediction_days:last_train_index + 31]  # Include enough days for predictions
    x_test = []
    for x in range(prediction_days, len(test_data)):
        x_test.append(test_data[x-prediction_days:x, 0])
    
    # Ensure there is data to make predictions
    if len(x_test) == 0:
        raise ValueError(f"Not enough data to form the x_test array. Adjust the range of test_data or prediction_days.")

    x_test = np.array(x_test).reshape((len(x_test), prediction_days, 1))

    # Predicting August prices
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    predictions[m] = predicted_price.flatten()

# Flatten all predictions and actual prices for min/max calculation
all_predicted = np.concatenate(list(predictions.values()))
actual_august_prices = data['Close'].values[last_train_index + 1:]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_august_prices, color='black', label='Actual August Prices')
plt.plot(predictions['simple_rnn'], label='Predicted Prices (Simple RNN)', color='blue')
plt.plot(predictions['lstm'], label='Predicted Prices (LSTM)', color='green')
plt.plot(predictions['bilstm'], label='Predicted Prices (BiLSTM)', color='red')

plt.title('AAPL August 2024 Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

# Adjust y-axis to fit all data
plt.ylim(min(min(actual_august_prices), all_predicted.min()) - 5, 
         max(max(actual_august_prices), all_predicted.max()) + 5)
plt.show()

# Print predictions to check values
for model, pred in predictions.items():
    print(f"{model} predictions: {pred}")


from sklearn.metrics import mean_squared_error

# Calculate and print RMSE for each model
for model, pred in predictions.items():
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_august_prices[:len(pred)], pred))
    print(f"{model} RMSE: {rmse:.4f}")
