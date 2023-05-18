import pandas as pd

# Read the data from the CSV files
events_df = pd.read_csv('./donyc_data/total_donyc_events.csv')
holidays_df = pd.read_csv('./nyc_holidays_data/holidays_data.csv')
weather_df = pd.read_csv('./weather_data/weather_data.csv')

# Convert date formats
events_df['date'] = pd.to_datetime(events_df['date'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'], format='%m/%d/%Y')
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Merge the dataframes based on the 'date' column
consolidated_df = events_df.merge(holidays_df, on='date').merge(weather_df, on='date')

# Save the consolidated dataframe as CSV
consolidated_df.to_csv('consolidated_signals.csv', index=False)
