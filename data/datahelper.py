import pandas as pd
import requests
from bs4 import BeautifulSoup
import os


def get_total_events_count(queried_text):
    """
    Retrieves the total number of events for each day from January 1, 2020 to June 1, 2023 from donyc.com.
    Returns a DataFrame with columns 'date' and 'total_events'.
    """

    # Prep dataframe for all applicable date ranges for which there will be total_{QUERIED_TEXT} count
    queried_start_date = pd.to_datetime('2020-01-01')
    queried_end_date = pd.to_datetime('2023-06-01')
    queried_date_range = pd.date_range(start=queried_start_date, end=queried_end_date)
    queried_date_range_df = pd.DataFrame({'date': queried_date_range})
    queried_date_range_df[f'total_{queried_text}'] = None

    # Set the base URL
    base_url = f'https://donyc.com/events/{{}}/{{}}/{{}}?page={{}}'

    # Verify that file exists
    csv_filename = f'donyc_{queried_text}.csv'
    if os.path.isfile(csv_filename):
        queried_date_range_df = pd.read_csv(csv_filename, parse_dates=['date'])

    # Loop through each date in the date range
    for i, row in queried_date_range_df.iterrows():

        date = row['date']

        # Check if the 'total_{queried_text}' column is not NaN, if yes then break out of the loop
        if not pd.isna(row[f'total_{queried_text}']):
            print(f'{date} is already populated')
            continue

        # Format the URL with the year, month, day, and page number
        year = date.year
        month = date.month
        day = date.day
        page_num = 1

        # Initialize the count to 0
        count = 0

        # Loop through each page of {queried_text} for the current date
        while True:
            # Make a request to the current page
            url = base_url.format(year, month, day, page_num)
            response = requests.get(url)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all div elements with a class attribute starting with 'ds-listing event-card'
            event_cards = soup.select(f'div[class^="ds-listing {queried_text}-card"]')

            # If no event cards are found, break out of the loop
            if not event_cards:
                break

            # Loop through each event card that matches the specified classes
            for card in event_cards:
                # Find the anchor tag with an href attribute that starts with '/{queried_text}/2023/3/1' and the specified classes
                anchor = card.find('a', href=lambda href: href and href.startswith(f'/events/{{}}/{{}}/{{}}'.format(year, month, day)), class_=f'ds-listing-{queried_text}-title url summary')
                if anchor:
                    count += 1

            # Increment the page number and update the URL
            print(page_num)
            page_num += 1

        # Set the count for the current date in the 'total_{queried_text}' column of the DataFrame
        queried_date_range_df.loc[i, f'total_{queried_text}'] = count
        queried_date_range_df.to_csv(csv_filename, index=False)
        print(queried_date_range_df.loc[i])


def get_total_categorial_events_count(queried_text):
    """
    Scrape https://donyc.com/ for the total number of categorial events for each date between
    January 1, 2020 and June 1, 2023 and saves to a file.
    """

    # Prepare dataframe for all applicable date ranges for which there will be total_{queried_text} count
    queried_start_date = pd.to_datetime('2020-01-01')
    queried_end_date = pd.to_datetime('2023-06-01')
    queried_date_range = pd.date_range(start=queried_start_date, end=queried_end_date)
    queried_date_range_df = pd.DataFrame({'date': queried_date_range})
    queried_date_range_df[f'total_{queried_text}'] = None

    # Set the base URL
    base_url = 'https://donyc.com/events/{}/{}/{}/{}?page={}'

    # Verify that file exists
    csv_filename = f'donyc_{queried_text}.csv'
    if os.path.isfile(csv_filename):
        queried_date_range_df = pd.read_csv(csv_filename, parse_dates=['date'])

    # Loop through each date in the date range
    for i, row in queried_date_range_df.iterrows():

        continue_for = False

        date = row['date']

        # Check if the 'total_{queried_text}' column is not NaN, if yes then break out of the loop
        if not pd.isna(row[f'total_{queried_text}']):
            print(f'{date} is already populated')
            continue

        # Format the URL with the year, month, day, and page number
        year = date.year
        month = date.month
        day = date.day
        page_num = 1

        # Initialize the count to 0
        count = 0

        # Loop through each page of {queried_text} for the current date
        while True:

            url = base_url.format(queried_text, year, month, day, page_num)

            # Make a request to the current page
            response = requests.get(url)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all div elements with a class attribute starting with 'ds-listing event-card'
            event_cards = soup.select(f'div[class^="ds-listing {queried_text}-card"]')

            # If no event cards are found, break out of the loop
            if not event_cards:
                break

            # Loop through each event card that matches the specified classes
            for card in event_cards:
                # Find the anchor tag with an href attribute that starts with '/events/{}/{}/{}/' and the specified classes
                anchor = card.find('a', href=lambda href: href and href.startswith(f'/events/{{}}/{{}}/{{}}/'.format(year, month, day)), class_=f'ds-listing-{queried_text}-title url summary')
                if anchor:
                    count += 1
                else:
                    continue_for = True
                    break

            # Increment the page number and update the URL
            print(page_num)
            page_num += 1

            if continue_for:
                break

        # Set the count for the current date in the 'total_{queried_text}' column of the DataFrame
        queried_date_range_df.loc[i, f'total_{queried_text}'] = count
        queried_date_range_df.to_csv(csv_filename, index=False)
        print(queried_date_range_df.loc[i])


def get_venue_events_bool(queried_text):
    '''
    Scrapes the donyc.com website for past and future events at Madison Square Garden
    and saves a CSV file with the dates for which events occurred.
    '''
    queried_text = 'madison-square-garden'
    csv_filename = f'donyc_{queried_text}.csv'

    # Prep dataframe for all applicable date ranges for which there will be total_{QUERIED_TEXT} count
    queried_start_date = pd.to_datetime('2020-01-01')
    queried_current_date = date.today()
    queried_end_date = pd.to_datetime('2023-06-01')

    queried_past_date_range = pd.date_range(start=queried_start_date, end=queried_current_date - timedelta(days=1))
    queried_past_date_df = pd.DataFrame({'date': queried_past_date_range})
    queried_past_date_df[f'{queried_text}_event_occurred'] = 0
    queried_past_date_df = queried_past_date_df[::-1]
    queried_past_date_df.reset_index(inplace=True, drop=True)

    queried_future_date_range = pd.date_range(start=queried_current_date, end=queried_end_date)
    queried_future_date_df = pd.DataFrame({'date': queried_future_date_range})
    queried_future_date_df[f'{queried_text}_event_occurred'] = 0

    # Set the base URL
    past_base_url = 'https://donyc.com/venues/{}/past_events?page={}'
    future_base_url = 'https://donyc.com/venues/{}?page={}'

    # Reminder, in the past loop, the "newer" Dates in code have lower date values
    page_num = 1
    logic_complete = False
    while True:
        url = past_base_url.format(queried_text, page_num)
        print(f'{url}')
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        event_cards = soup.select('div[class^="ds-listing event-card"]')

        for card in event_cards:
            href = card.find('a')['href']
            year, month, day = [int(x) for x in href.split('/')[2:5] if x.isdigit()]
            event_date = pd.to_datetime(date(year, month, day))
            print(event_date)

            if event_date < queried_start_date.date():
                logic_complete = True
                break

            queried_past_date_df.loc[queried_past_date_df['date'] == event_date, f'{queried_text}_event_occurred'] = 1
            pd.concat([queried_past_date_df, queried_future_date_df]).sort_values('date', ascending=True).to_csv(csv_filename, index=False)

        if logic_complete:
            break

        # one page is complete, run the next page
        print(page_num)
        page_num += 1

    # Reminder, in the future loop, the "newer" Dates in code have higher date values
    page_num = 1
    logic_complete = False
    while True:
        url = future_base_url.format(queried_text, page_num)
        print(f'{url}')
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        event_cards = soup.select('div[class^="ds-listing event-card"]')

        for card in event_cards:
            href = card.find('a')['href']
            year, month, day = [int(x) for x in href.split('/')[2:5] if x.isdigit()]
            event_date = pd.to_datetime(date(year, month, day))
            print(event_date)

            if event_date > queried_end_date.date():
                logic_complete = True
                break

            queried_future_date_df.loc[queried_future_date_df['date'] == event_date, f'{queried_text}_event_occurred'] = 1
            pd.concat([queried_past_date_df, queried_future_date_df]).sort_values('date', ascending=True).to_csv(csv_filename, index=False)

        if logic_complete:
            break

        # one page is complete, run the next page
        print(page_num)
        page_num += 1
