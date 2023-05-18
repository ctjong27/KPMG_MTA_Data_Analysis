# KPMG_MTA
KPMG MTA Project

Key objective: To identify use case of models that predicts New York City's MTA Subway transit ridership.


# Data
./data/ : Houses the code used to retrieve data as well as consolidated_signals.csv that feeds into models

Each of the child folders contain an aggregated daily data from respective sources.

./data/data_consolidator.py : Consolidates data from each of the child folders into consolidated_signals.csv

./data/ridership_data/ : MTA Daily Ridership Data retrieved from data.ny.gov

./data/donyc_data/ : Houses the data retrieved from donyc.com

./data/nyc_holidays_data/ : Houses the holidays retrieved from nyc.com

./data/weather_data/ : Houses the weather data retrieved from world-weather.info


# Modeling
./modeling/ : Contains codes used to tune parameters and generate regression reports and plots

./modeling/4_25_prophet_parameter_tuning.py : A work-in-progress of optimizing parameters for the prophet

./modeling/prophet_parameter_tuning.py : Optimized prophet model with proper parameters

./modeling/prophet_featureimportance.py : From the root directory, generates report of Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and feature importance for the forecatsed date range


# Misc Files
License : MIT License

requirements.txt : Contains a list of dependences required to run code properly

./_archives/ : Contains work-in-progress files used in various portions of the project