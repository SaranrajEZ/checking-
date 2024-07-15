import pandas as pd
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

# Define the start and end dates
start = dt.datetime(2021, 6, 1)
end = dt.datetime(2022, 1, 5)
symbol = 'AXISBANK.NS'

# Fetch stock data
stk_data = yf.download(symbol, start=start, end=end)
stk_data = stk_data[['Open', 'High', 'Low', 'Close']]

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(stk_data)
data_normalized = pd.DataFrame(data_normalized, columns=["Open", "High", "Low", "Close"])

# Split data into train and test sets
test_obs = 29
train = data_normalized[:-test_obs]
test = data_normalized[-test_obs:]

# Model selection using AIC and BIC
warnings.filterwarnings("ignore")
for i in range(1, 11):
    model = VAR(train)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()

# Fit the selected model
selected_order = model.select_order(maxlags=12).selected_orders["aic"]
result = model.fit(selected_order)

# Forecasting
lagged_values = train.values[-selected_order:]
forecast = result.forecast(y=lagged_values, steps=test_obs)
forecast_df = pd.DataFrame(forecast, columns=["Open", "High", "Low", "Close"])

# Calculate performance metrics
rmse = mean_squared_error(test, forecast_df, squared=False)
mape = mean_absolute_percentage_error(test, forecast_df)

# Display performance metrics
performance = {
    "Model": ["VAR"],
    "RMSE": [rmse],
    "MAPE": [mape],
    "Lag": [selected_order],
    "Test": [test_obs]
}
perf = pd.DataFrame(performance)
print(perf)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.plot(test["Close"].values, color='blue', label="Actual")
plt.plot(forecast_df["Close"].values, color='green', label="Predicted")
plt.title("AXISBANK-Close-AR-Norm")
plt.xlabel("Days")
plt.ylabel("Prices")
plt.legend()
plt.show()

# Inverse transform to original scale
actual_stock_price_test_ori = scaler.inverse_transform(test)
actual_stock_price_test_ori_df = pd.DataFrame(actual_stock_price_test_ori, columns=["Open", "High", "Low", "Close"])

predicted_stock_price_test_ori = scaler.inverse_transform(forecast_df)
predicted_stock_price_test_ori_df = pd.DataFrame(predicted_stock_price_test_ori, columns=["Open", "High", "Low", "Close"])

# Plotting actual vs. predicted values on original scale
plt.figure(figsize=(10, 5))
plt.plot(actual_stock_price_test_ori_df["Close"], color='blue', label="Actual")
plt.plot(predicted_stock_price_test_ori_df["Close"], color='green', label="Predicted")
plt.title("AXISBANK-Close-AR-Norm")
plt.xlabel("Days")
plt.ylabel("Prices")
plt.legend()
plt.show()

# Calculate performance metrics on original scale
rmse_ori = mean_squared_error(actual_stock_price_test_ori_df["Close"], predicted_stock_price_test_ori_df["Close"], squared=False)
mape_ori = mean_absolute_percentage_error(actual_stock_price_test_ori_df["Close"], predicted_stock_price_test_ori_df["Close"])
print("RMSE-Testset (Original):", rmse_ori)
print("MAPE-Testset (Original):", mape_ori)

# Forecast future prices
forecast_steps = 3  # Adjust as needed
forecast_future = result.forecast(result.endog, steps=forecast_steps)
forecast_future_df = pd.DataFrame(forecast_future, columns=["Open", "High", "Low", "Close"])

# Inverse transform future forecast to original scale
forecast_future_prices = scaler.inverse_transform(forecast_future_df)
forecast_future_prices_df = pd.DataFrame(forecast_future_prices, columns=["Open", "High", "Low", "Close"])

# Save forecasted prices
forecast_future_prices_df.to_csv("CloseVARForecast.csv", index=False)

# Plotting actual vs. forecasted future prices
plt.figure(figsize=(10, 5))
plt.plot(actual_stock_price_test_ori_df["Close"].iloc[-forecast_steps:], color='blue', label="Actual")
plt.plot(forecast_future_prices_df["Close"], color='red', label="Forecasted")
plt.title("AXISBANK-Close-VAR-Forecast")
plt.xlabel("Days")
plt.ylabel("Prices")
plt.legend()
plt.show()
    