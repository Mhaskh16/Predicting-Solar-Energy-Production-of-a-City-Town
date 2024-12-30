import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf ,pacf
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

# Load the data
file_path = r'D:\HU Documents\Project\Data For Project\5yeartemp.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data.set_index('Date', inplace=True)

# Set frequency to daily
data = data.asfreq('D')

# Split data into maximum and minimum temperature
tempmax = data['tempmax']
tempmin = data['tempmin']

tempmax.head(10)

plt.xlabel("Date")
plt.ylabel("temp")
plt.plot(tempmax, label='TempMax', color='blue')
plt.plot(tempmin, label='TempMin', color='green')
plt.legend()

# Check for stationarity using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary. Consider differencing.")

check_stationarity(tempmax)

from statsmodels.tsa.stattools import acf ,pacf
tempmax_lagAcf = acf(tempmax, nlags=87)
tempmax_lagPacf = pacf(tempmax, nlags=20, method='ols')

#plot ACF
plt.subplot(121)
plt.plot(tempmax_lagAcf)
plt.axhline(y=0, linestyle='--', color='grey')
plt.axhline(y=-1.96/np.sqrt(len(tempmax)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(tempmax)),linestyle='--',color='grey')
plt.title('Auto Correlation Function')

#plot PACF
plt.subplot(122)
plt.plot(tempmax_lagPacf)
plt.axhline(y=0, linestyle='--', color='grey')
plt.axhline(y=-1.96/np.sqrt(len(tempmax)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(tempmax)),linestyle='--',color='grey')
plt.title('Partial Auto Correlation Function')

# Define a function to fit ARIMA and forecast
def arima_forecast(series, order, steps):
    # Fit ARIMA model
    model = ARIMA(series, order=order)
    results = model.fit()
    
    forecast = results.get_forecast(steps=steps)
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    conf_int = forecast.conf_int()
    
    return forecast_series, conf_int

# Define ARIMA parameters (order=(p, d, q))
arima_order = (87, 0, 10)

# Forecast for December 2024 (31 days)
forecast_steps = 31
forecast_max, conf_int_max = arima_forecast(tempmax, arima_order, steps=forecast_steps)

plt.plot(forecast_max, color='Red')
plt.plot(conf_int_max, color='Green')

# Combine forecasts into a DataFrame
forecast_df = pd.DataFrame({
    'Date': forecast_max.index,
    'forecast_tempmax': forecast_max.values,
    #'forecast_tempmin': forecast_min.values,
    'tempmax_lower': conf_int_max.iloc[:, 0].values,
   'tempmax_upper': conf_int_max.iloc[:, 1].values,
    #'tempmin_lower': conf_int_min.iloc[:, 0].values,
    #'tempmin_upper': conf_int_min.iloc[:, 1].values
})

# Save the forecast to the specified directory
output_path = r'D:\HU Documents\Project\Data For Project\forecast_dec_2024.xlsx'
forecast_df.to_excel(output_path, index=False)
print(f"Forecast for December 2024 saved successfully to {output_path}")

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(tempmax[-365:], label='Observed TempMax', color='blue')
plt.plot(forecast_max, label='Forecast TempMax', color='red', linestyle='--')
plt.fill_between(forecast_max.index, conf_int_max.iloc[:, 0], conf_int_max.iloc[:, 1], color='red', alpha=0.2)

