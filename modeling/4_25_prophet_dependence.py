
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import seaborn as sns

#data = pd.read_csv('/content/Simplified All New Signals_weather holiday.csv')
data = pd.read_csv('../data/consolidated_signals.csv')
data['date'] = pd.to_datetime(data['date'])
start_date = '2020-04-20'
end_date = '2023-03-16'
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
data.tail()

data = pd.get_dummies(data, prefix=['weather'], columns=['weather'])
data

#rename for prophet requirement
data = data.rename(columns={"date": "ds", "ridership": "y"})

data.columns

data

#check null values
print(data.isna().sum())

data = data.dropna()
data.info()

'''
fold1 = data[(data['ds'] >= start_date) & (data['ds'] <= "2020-09-25")]
fold2 = data[(data['ds'] >= start_date) & (data['ds'] <= "2021-03-24")]
fold3 = data[(data['ds'] >= start_date) & (data['ds'] <= "2021-09-20")]
fold4 = data[(data['ds'] >= start_date) & (data['ds'] <= "2022-03-19")]
fold5 = data[(data['ds'] >= start_date) & (data['ds'] <= "2022-09-15")]
fold6 = data[(data['ds'] >= start_date) & (data['ds'] <= "2023-03-02")]
'''

data.tail()

fold1 = data[(data['ds'] >= start_date) & (data['ds'] <= "2020-09-11")]
fold2 = data[(data['ds'] >= start_date) & (data['ds'] <= "2021-03-10")]
fold3 = data[(data['ds'] >= start_date) & (data['ds'] <= "2021-09-06")]
fold4 = data[(data['ds'] >= start_date) & (data['ds'] <= "2022-03-05")]
fold5 = data[(data['ds'] >= start_date) & (data['ds'] <= "2022-09-01")]
fold6 = data[(data['ds'] >= start_date) & (data['ds'] <= "2023-02-16")]
fold7 = data[(data['ds'] >= "2022-02-01") & (data['ds'] <= "2023-03-02")]
testing = data[(data['ds'] >= "2023-03-03") & (data['ds'] <= "2023-03-16")]

fold7.columns

"""# Event related high relevant sigals"""

event = fold7[['ds','y','total_comedy','total_events','total_film-screenings','total_music']]

m1 = Prophet()
for column in event.columns:
  if column !='ds' and column != 'y':
    m1.add_regressor(column)

m1.fit(event)
forecast1 = m1.predict(testing)

# plot only the forecast
fig = m1.plot(forecast1, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([testing['ds'].min(), testing['ds'].max()])
dates = testing['ds']
plt.plot(dates,testing['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

test_data = testing
y_true = test_data['y'].values
y_pred1 = forecast1['yhat'].values
absolute_percentage_error = np.abs((y_true - y_pred1) / y_true)
mean_absolute_percentage_error = np.mean(absolute_percentage_error) * 100
mean_absolute_percentage_error

"""# Event and temp"""

eventemp = fold7[['ds','y','high_temp','low_temp','total_comedy','total_events','total_film-screenings','total_music']]

m2 = Prophet()
for column in eventemp.columns:
  if column !='ds' and column != 'y':
    m2.add_regressor(column)

m2.fit(eventemp)
forecast2 = m2.predict(testing)

# plot only the forecast
fig = m2.plot(forecast2, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([testing['ds'].min(), testing['ds'].max()])
dates = testing['ds']
plt.plot(dates,testing['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

test_data = testing
y_true = test_data['y'].values
y_pred1 = forecast2['yhat'].values
absolute_percentage_error = np.abs((y_true - y_pred1) / y_true)
mean_absolute_percentage_error = np.mean(absolute_percentage_error) * 100
mean_absolute_percentage_error

"""# using all signals"""

# Separate the data into training and testing sets
train_data = fold7
test_data = testing

# Initialize the Prophet model
m = Prophet()

# Add the regressors to the model for both training and testing sets
for column in fold7.columns:
    if column != 'ds' and column != 'y':
        m.add_regressor(column)

# Fit the model to the training set
m.fit(train_data)

# Make predictions for the testing set
#future = m.make_future_dataframe(periods=len(test_data))
forecast = m.predict(testing)

# plot only the forecast
fig = m.plot(forecast, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([testing['ds'].min(), testing['ds'].max()])
dates = testing['ds']
plt.plot(dates,testing['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

y_true = test_data['y'].values
y_pred = forecast['yhat'].values
absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
mean_absolute_percentage_error = np.mean(absolute_percentage_error) * 100
mean_absolute_percentage_error

"""#grid search for fold 7"""

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

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(fold7)  # Fit model with given params
    forecast = m.predict(testing)
    score = np.mean(np.abs((testing['y'].values - forecast['yhat'].values)/testing['y'].values))*100
    mapes.append(score)

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mapes
print(tuning_results)

tuning_results.sort_values('mape', ascending=True)

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

# Create a list of colors for the bars
colors = ['red', 'blue', 'green', 'orange']

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10,8))
y_pos = np.arange(len(tuning_results.index))
ax.barh(y_pos, tuning_results['mape'], color=colors, alpha=0.8)

# Add labels and a title
ax.set_yticks(y_pos)
ax.set_yticklabels(tuning_results.index)
ax.set_xlabel('MAPE')
ax.set_title('MAPE values by parameter settings')

# Show the plot
plt.show()

"""#grid search for events

"""

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 10],
    'seasonality_mode':['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes1 = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m1 = Prophet(**params).fit(event)  # Fit model with given params
    forecast1 = m1.predict(testing)
    score1 = np.mean(np.abs((testing['y'].values - forecast1['yhat'].values)/testing['y'].values))*100
    mapes1.append(score1)

# Find the best parameters
tuning_results1 = pd.DataFrame(all_params)
tuning_results1['mape1'] = mapes1
print(tuning_results1)




param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 10],
    'seasonality_mode':['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(fold7)  # Fit model with given params
    forecast = m.predict(testing)
    score = np.mean(np.abs((testing['y'].values - forecast['yhat'].values)/testing['y'].values))*100
    mapes.append(score)

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mapes
print(tuning_results)

forecast1

forecast

score1

tuning_results1.sort_values('mape1',ascending = True)

mapes1

mapes

"""#grid search for events and temp"""

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 10],
    'seasonality_mode':['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mapes2 = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m2 = Prophet(**params).fit(eventemp)  # Fit model with given params
    forecast2 = m2.predict(testing)
    score2 = np.mean(np.abs((testing['y'].values - forecast2['yhat'].values)/testing['y'].values))*100
    mapes2.append(score2)

# Find the best parameters
tuning_results2 = pd.DataFrame(all_params)
tuning_results2['mape2'] = mapes2
print(tuning_results2)

tuning_results2.sort_values('mape2',ascending = True)



forecast = m.predict(testing)

# plot only the forecast
fig = m.plot(forecast, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([testing['ds'].min(), testing['ds'].max()])
dates = testing['ds']
plt.plot(dates,testing['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

y_true = test_data['y'].values
y_pred = forecast['yhat'].values
absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
mean_absolute_percentage_error = np.mean(absolute_percentage_error) * 100
mean_absolute_percentage_error

"""# event """



"""# Fold 1"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold1

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params
    print (best_params)

# Fit final model with best hyperparameters
m = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m.fit(df)

validation_start = '2023-03-03'
validation_end = '2023-03-16'
validation_data = data[(data['ds'] >= validation_start) & (data['ds'] <= validation_end)]
validation_data = validation_data.reset_index()

validation_data

forecast = m.predict(validation_data)

forecast

forecast['actual'] = validation_data['y']

validation_data['y']

forecast['actual']

# plot only the forecast
fig = m.plot(forecast, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

#fold 1
ape = np.abs((forecast['actual'] - forecast['trend']) / forecast['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

"""#fold 2"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold2

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m2 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m2.fit(df)

forecast2 = m2.predict(validation_data)
forecast2['actual'] = validation_data['y']

# plot only the forecast
fig = m2.plot(forecast2, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 2
ape = np.abs((forecast2['actual'] - forecast2['trend']) / forecast2['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

"""#fold 3"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold3

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m3 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m3.fit(df)

forecast3 = m3.predict(validation_data)
forecast3['actual'] = validation_data['y']

# plot only the forecast
fig = m3.plot(forecast3, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 3
ape = np.abs((forecast3['actual'] - forecast3['trend']) / forecast3['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

"""#fold 4"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold4

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m4 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m4.fit(df)

forecast4 = m4.predict(validation_data)
forecast4['actual'] = validation_data['y']

# plot only the forecast
fig = m4.plot(forecast, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 4
ape = np.abs((forecast4['actual'] - forecast4['trend']) / forecast4['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

"""#fold 5"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold5

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m5 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m5.fit(df)

forecast5 = m5.predict(validation_data)
forecast5['actual'] = validation_data['y']

# plot only the forecast
fig = m5.plot(forecast, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 4
ape = np.abs((forecast5['actual'] - forecast5['trend']) / forecast5['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

"""#fold 6"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold6

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m6 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m6.fit(df)

forecast6 = m6.predict(validation_data)
forecast6['actual'] = validation_data['y']

# plot only the forecast
fig = m6.plot(forecast6, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 6
ape = np.abs((forecast6['actual'] - forecast6['trend']) / forecast6['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)



"""#fold 7"""

from sklearn.model_selection import ParameterGrid
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd

# Define parameter grid
param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Load data
df = fold7

# Perform grid search CV
best_score = np.inf
for params in ParameterGrid(param_grid):
    model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'])
    model.fit(df)
    forecast = model.predict(df)
    score = mean_squared_error(forecast['yhat'], df['y'])
    if score < best_score:
        best_score = score
        best_params = params

# Fit final model with best hyperparameters
m7 = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                      seasonality_mode=best_params['seasonality_mode'])
m7.fit(df)

forecast7 = m7.predict(validation_data)
forecast7['actual'] = validation_data['y']

validation_data

# plot only the forecast
fig = m7.plot(forecast7, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 7
ape = np.abs((forecast7['actual'] - forecast7['trend']) / forecast7['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)

#fold 7
#multivariate modeling
m7 = Prophet()

# Add the regressors to the model
# assume that you have already defined the Prophet model m and have loaded the data into the DataFrame df
for column in fold4.columns:
    if column != 'ds' and column != 'y':
        m7.add_regressor(column)

# Fit the model
m7.fit(fold7)

forecast7 = m7.predict(validation_data)
forecast7['actual'] = validation_data['y']

# plot only the forecast
fig = m7.plot(forecast6, xlabel='Date', ylabel='Value',figsize=(10,4))
fig.gca().set_xlim([validation_data['ds'].min(), validation_data['ds'].max()])
dates = validation_data['ds']
plt.plot(dates,validation_data['y'],label = 'Actual',color = 'red')

# display the plot
plt.legend()
plt.show()

#fold 7
ape = np.abs((forecast7['actual'] - forecast7['trend']) / forecast7['trend'])
mape = np.mean(ape) * 100

print('MAPE: %.3f%%' % mape)