import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

"""# Run Data Analysis

## Data Retrieval, metric result for MAE & MAPE, and feature importance plot
"""

# Load the data
data = pd.read_csv('./data/consolidated_signals.csv')
data['date'] = pd.to_datetime(data['date'])
start_date = '2020-04-20'
end_date = '2023-03-16'
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Perform one-hot encoding for 'weather' column
data = pd.get_dummies(data, prefix=['weather'], columns=['weather'])

# Rename columns for Prophet requirement
data = data.rename(columns={"date": "ds", "ridership": "y"})

# Drop any rows with null values
data = data.dropna()

# Create training and testing sets
fold7 = data[(data['ds'] >= "2022-02-01") & (data['ds'] <= "2023-03-02")]
testing = data[(data['ds'] >= "2023-03-03") & (data['ds'] <= "2023-03-16")]

# Initialize the Prophet model
m = Prophet(changepoint_prior_scale=0.010, seasonality_prior_scale=10, holidays_prior_scale=10, seasonality_mode='multiplicative')

# Add the regressors to the model for both training and testing sets
for column in fold7.columns:
    if column != 'ds' and column != 'y':
        m.add_regressor(column)

# Fit the model to the training set
m.fit(fold7)

# Make predictions for the testing set
forecast = m.predict(testing)

# Create a stacked area plot for cumulative contributions of regressors
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate through each regressor and plot its cumulative contribution to the target variable
for column in fold7.columns:
    if column != 'ds' and column != 'y':
        ax.fill_between(testing['ds'], forecast[column].cumsum(), label=column)

ax.set_title('Cumulative Contribution of Regressors to Target Variable')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Contribution')
ax.legend(loc='upper left') # Update legend position
plt.tight_layout() # Add tight layout to prevent overlapping
plt.show()

# Plot the actual values and forecast
fig = m.plot(forecast, xlabel='Date', ylabel='Value', figsize=(12, 6)) # Increase plot size
fig.gca().set_xlim([testing['ds'].min(), testing['ds'].max()])
dates = testing['ds']
plt.plot(dates, testing['y'], label='Actual', color='red')
plt.legend()
plt.tight_layout() # Add tight layout to prevent overlapping
plt.show()

# Calculate Mean Absolute Percentage Error (MAPE)
y_true = testing['y'].values
y_pred = forecast['yhat'].values
absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
mean_absolute_percentage_error = np.mean(absolute_percentage_error) * 100

"""## Forecast components plots"""

fig = m.plot_components(forecast)
plt.show()

"""## Contributions of variables for each date"""

# Create an empty dataframe to store the contributions
contributions = pd.DataFrame()

# Iterate through each regressor and calculate its contribution for each time period
for column in fold7.columns:
    if column != 'ds' and column != 'y':
        contributions[column] = forecast[column]

# Add the date column from the testing dataframe to the contributions dataframe
contributions['date'] = testing['ds'].values

# Set the date column as the index of the contributions dataframe
contributions.set_index('date', inplace=True)

contributions

"""## Contributions of regressor for each date"""

import matplotlib.pyplot as plt

# Assuming you have a fitted Prophet model named 'm' with extra regressors
# Access the regressor data from the testing set
regressor_data = forecast[['ds'] + list(m.extra_regressors.keys())]

# Determine the number of rows and columns for subplots
n_rows = (len(m.extra_regressors) + 1) // 2
n_cols = 2

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
fig.tight_layout(pad=5)

# Loop through each regressor and plot its time series data in a subplot
for i, regressor_name in enumerate(m.extra_regressors.keys()):
    row_idx = i // n_cols
    col_idx = i % n_cols
    ax = axes[row_idx, col_idx]
    ax.plot(regressor_data['ds'], regressor_data[regressor_name])
    ax.set_title(f'Time Series of Regressor: {regressor_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

# Remove any unused subplots
if len(m.extra_regressors) % 2 != 0:
    axes[-1, -1].remove()

plt.show()

import matplotlib.pyplot as plt

# Assuming you have a fitted Prophet model named 'm' with extra regressors
# Access the regressor data
regressor_data = m.history

# Extract the names of the regressors
regressor_names = m.extra_regressors.keys()

# Determine the number of rows and columns for subplots
n_rows = (len(regressor_names) + 1) // 2
n_cols = 2

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
fig.tight_layout(pad=5)

# Loop through each regressor and plot its time series data in a subplot
for i, regressor_name in enumerate(regressor_names):
    row_idx = i // n_cols
    col_idx = i % n_cols
    ax = axes[row_idx, col_idx]
    ax.plot(regressor_data['ds'], regressor_data[regressor_name])
    ax.set_title(f'Time Series of Regressor: {regressor_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

# Remove any unused subplots
if len(regressor_names) % 2 != 0:
    axes[-1, -1].remove()

plt.show()

"""# Results"""

print("")
print("########################################################################")
print("")
print(f"absolute_percentage_error: {absolute_percentage_error}")
print(f"mean_absolute_percentage_error: {mean_absolute_percentage_error}")



# Find the highest value in each column
max_values = contributions.max()

# Iterate over each column and find the key and value for the highest value
print("")
print("########################################################################")
print("")
print("Highest Value & Date pairs of feature importance in the forecasted dates")
result = []
for column in contributions.columns:
    key = contributions[column].idxmax()
    value = contributions.loc[key, column]
    result.append(f"The highest of '{column}'  with a value of '{value}' on day '{key}'")

# Print the formatted text
for text in result:
    print(text)

    
    
# Compute the sum of each column
column_sums = contributions.sum()

# Sort the column sums in descending order
sorted_sums = column_sums.sort_values(ascending=False)

# Print the results in descending order
print("")
print("########################################################################")
print("")
print("Decending value of aggregated sum of feature importance in the forecasted dates")
for column, value in sorted_sums.items():
    print(f"Sum of '{column}' has a lifetime significance of {value}.")