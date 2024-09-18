import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load dataset for regression
data = pd.read_csv('regression_data.csv')
X = data[['independent_variable1', 'independent_variable2']]
y = data['dependent_variable']

# Add constant term for intercept
X = sm.add_constant(X)

# Multiple Linear Regression
linear_model = sm.OLS(y, X).fit()
linear_predictions = linear_model.predict(X)
print("Linear Regression Summary:")
print(linear_model.summary())

# Logistic Regression
y_binary = data['binary_outcome']
logit_model = sm.Logit(y_binary, X).fit()
logit_predictions = logit_model.predict(X)
print("\nLogistic Regression Summary:")
print(logit_model.summary())

# Hypothesis Testing: T-test
t_test_result = sm.stats.ttest_ind(data['group1'], data['group2'])
print("\nT-test Result:")
print(f"T-statistic: {t_test_result[0]}, p-value: {t_test_result[1]}")

# Load dataset for time series analysis
ts_data = pd.read_csv('timeseries_data.csv', index_col='date', parse_dates=True)
ts = ts_data['value']

# Plot time series data
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Time Series Data')
plt.show()

# Autocorrelation and Partial Autocorrelation plots
plot_acf(ts)
plt.show()
plot_pacf(ts)
plt.show()

# ARIMA Model
arima_model = ARIMA(ts, order=(1, 1, 1)).fit()
arima_predictions = arima_model.predict(start=len(ts), end=len(ts) + 10, typ='levels')
print("\nARIMA Model Summary:")
print(arima_model.summary())

# Plot ARIMA predictions
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Original')
plt.plot(arima_predictions, label='Predicted', color='red')
plt.legend()
plt.title('ARIMA Predictions')
plt.show()
