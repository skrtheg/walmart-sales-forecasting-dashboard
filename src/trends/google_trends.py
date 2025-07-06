from pytrends.request import TrendReq
import pandas as pd

def fetch_weekly_trends(keywords, start_date, end_date, geo='US'):
    """
    Fetch weekly Google Trends data for a list of keywords between start_date and end_date.
    Returns a DataFrame with columns: Date, <keyword1>, <keyword2>, ...
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    all_trends = []
    for kw in keywords:
        pytrends.build_payload([kw], timeframe=f'{start_date} {end_date}', geo=geo)
        df = pytrends.interest_over_time().reset_index()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        df = df.rename(columns={kw: f'Trends_{kw.replace(" ", "_")}'})
        all_trends.append(df)
    # Merge all trends on Date
    result = all_trends[0]
    for df in all_trends[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    result = result.rename(columns={'date': 'Date'})
    result['Date'] = pd.to_datetime(result['Date'])
    return result
