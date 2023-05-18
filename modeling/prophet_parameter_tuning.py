import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import seaborn as sns

data = pd.read_csv('./data/consolidated_signals.csv')
data['date'] = pd.to_datetime(data['date'])
start_date = '2020-04-20'
end_date = '2023-03-16'
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
data = pd.get_dummies(data, prefix=['weather'], columns=['weather'])
data
#rename for prophet requirement
data = data.rename(columns={"date": "ds", "ridership": "y"})

#check null values
print(data.isna().sum())

data = data.dropna()
data.info()

#slice the training period: 02.01.2022 – 03.02.2023 
fold7 = data[(data['ds'] >= "2022-02-01") & (data['ds'] <= "2023-03-02")]
#slice the testing period: 3.03.2023 – 03.16.2023
testing = data[(data['ds'] >= "2023-03-03") & (data['ds'] <= "2023-03-16")]

"""#grid search for smallest mape value"""

import itertools

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 10],
    'seasonality_mode':['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes = []  # Store the RMSEs for each params here
forecast_list = []

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(fold7)  # Fit model with given params
    forecast = m.predict(testing)
    score = np.mean(np.abs((testing['y'].values - forecast['yhat'].values)/testing['y'].values))*100
    forecast['score'] = score
    mapes.append(score)
    forecast_list.append(forecast)

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mapes
print(tuning_results)

#get all "mape" values from grid search
tuning_results.sort_values('mape', ascending=True)

min_score = min([df['score'].min() for df in forecast_list])
#get the most accurate prediction
acc_pre = pd.DataFrame([df for df in forecast_list if df['score'].min() == min_score][0])
acc_pre

"""Plot the prediction with lowest MAPE value: 4.24%"""

plt.figure(figsize=(12, 6))
# Plot only the forecast
plt.plot(dates, acc_pre['yhat'], label='Prediction', color='blue', linewidth=2)
plt.plot(dates, testing['y'], label='Actual', color='red', linewidth=2)

# Set the title and axis labels
plt.title('Actual vs. Predicted Ridership', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Ridership', fontsize=14)

# Add a grid and legend
plt.grid(True)
plt.legend(loc='upper left', fontsize=12)

# Customize the x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

"""Get the distribution of MAPE values for each grid search combination"""

# Sort the dataframe by 'mape'
tuning_results_sorted = tuning_results.sort_values(by='mape',ascending =True).reset_index()

# Create a horizontal bar chart
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='mape', y=tuning_results_sorted.index, data=tuning_results_sorted, ax=ax, palette='colorblind',orient='h')

# Add labels and a title
ax.set_xlabel('MAPE')
ax.set_ylabel('')
ax.set_title('MAPE values by parameter settings')

# Show the plot
plt.show()