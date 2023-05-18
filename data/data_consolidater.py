import pandas as pd

# Read the data from the CSV files
events_df = pd.read_csv('./donyc_data/total_donyc_events.csv')
holidays_df = pd.read_csv('./nyc_holidays_data/holidays_data.csv')

# Convert date formats
events_df['date'] = pd.to_datetime(events_df['date'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'], format='%m/%d/%Y')

# Merge the dataframes based on the 'date' column
consolidated_df = events_df.merge(holidays_df, on='date')

# Save the consolidated dataframe as CSV
consolidated_df.to_csv('consolidated_signals.csv', index=False)
