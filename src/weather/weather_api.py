import requests
import pandas as pd
from datetime import datetime

# Example: Open-Meteo API for historical weather data (free, no API key required)
# Docs: https://open-meteo.com/en/docs

CITIES = {
    'Bentonville': {'lat': 36.3729, 'lon': -94.2088},
    'Dallas': {'lat': 32.7767, 'lon': -96.7970},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298},
    'Atlanta': {'lat': 33.7490, 'lon': -84.3880},
    # Add more Walmart cities as needed
}

def fetch_weather(city, start_date, end_date):
    """
    Fetch daily avg temperature and precipitation for a city between start_date and end_date.
    Returns a DataFrame with columns: Date, Avg_Temperature, Precipitation
    """
    coords = CITIES[city]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={coords['lat']}&longitude={coords['lon']}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=America/Chicago"
    )
    resp = requests.get(url)
    data = resp.json()['daily']
    df = pd.DataFrame({
        'Date': data['time'],
        'Avg_Temperature': [(tmax + tmin)/2 for tmax, tmin in zip(data['temperature_2m_max'], data['temperature_2m_min'])],
        'Precipitation': data['precipitation_sum']
    })
    df['Date'] = pd.to_datetime(df['Date'])
    df['City'] = city
    return df

def fetch_weather_for_all_cities(start_date, end_date):
    dfs = [fetch_weather(city, start_date, end_date) for city in CITIES]
    return pd.concat(dfs, ignore_index=True)
