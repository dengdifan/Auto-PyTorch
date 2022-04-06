# Seasonality values corresponding with the frequencies: minutely, 10_minutes, half_hourly, hourly, daily, weekly, monthly, quarterly and yearly
# Consider multiple seasonalities for frequencies less than daily

# The name of the column containing time series values after loading data from the .tsf file into a dataframe
VALUE_COL_NAME = "series_value"

# The name of the column containing timestamps after loading data from the .tsf file into a dataframe
TIME_COL_NAME = "start_timestamp"

BUDGET_TYPES = ['epochs', 'resolution', 'num_seq', 'num_sample_per_seq']

SEASONALITY_MAP = {
    "minutely": [1440, 10080, 525960],
    "10_minutes": [144, 1008, 52596],
    "half_hourly": [48, 336, 17532],
    "hourly": [24, 168, 8766],
    "daily": 7,
    "weekly": 365.25 / 7,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

# Frequencies used by GluonTS framework
FREQUENCY_MAP = {
    "minutely": "1min",
    "10_minutes": "10min",
    "half_hourly": "30min",
    "hourly": "1H",
    "daily": "1D",
    "weekly": "1W",
    "monthly": "1M",
    "quarterly": "1Q",
    "yearly": "1Y"
}

DATASETS = {"m1_yearly": ("m1_yearly_dataset.tsf", None, False),
            "m1_quarterly": ("m1_quarterly_dataset.tsf", None, False),
            "m1_monthly": ("m1_monthly_dataset.tsf", None, False),
            "m3_yearly": ("m3_yearly_dataset.tsf", None, False),
            "m3_quarterly": ("m3_quarterly_dataset.tsf", None, False),
            "m3_monthly": ("m3_monthly_dataset.tsf", None, False),
            "m3_other": ("m3_other_dataset.tsf", None, False),
            "m4_quarterly": ("m4_quarterly_dataset.tsf", None, False),
            "m4_monthly": ("m4_monthly_dataset.tsf", None, False),
            "m4_weekly": ("m4_weekly_dataset.tsf", None, False),
            "m4_daily": ("m4_daily_dataset.tsf", None, False),
            "m4_hourly": ("m4_hourly_dataset.tsf", None, False),
            "m4_yearly": ("m4_yearly_dataset.tsf", None, False),
            "tourism_yearly": ("tourism_yearly_dataset.tsf", None, False),
            "tourism_quarterly": ("tourism_quarterly_dataset.tsf", None, False),
            "tourism_monthly": ("tourism_monthly_dataset.tsf", None, False),
            #"cif_2016": ("cif_2016_dataset.tsf", 6, False),
            #"london_smart_meters": ("london_smart_meters_dataset_without_missing_values.tsf", 336, False),
            "aus_elecdemand": ("australian_electricity_demand_dataset.tsf", 336, False),
            "dominick": ("dominick_dataset.tsf", 8, False),
            "bitcoin": ("bitcoin_dataset_without_missing_values.tsf", 30, False),
            "melbourne_pedestrian_counts": ("pedestrian_counts_dataset.tsf", 24, True),
            "vehicle_trips": ("vehicle_trips_dataset_without_missing_values.tsf", 30, True),
            "kdd_cup": ("kdd_cup_2018_dataset_without_missing_values.tsf", 168, False),
            "weather": ("weather_dataset.tsf", 30, False),
            "nn5_daily": ("nn5_daily_dataset_without_missing_values.tsf", None, False),
            "nn5_weekly": ("nn5_weekly_dataset.tsf", 8, False),

            "car_parts": ("car_parts_dataset_without_missing_values.tsf", 12, True),
            "hospital": ("hospital_dataset.tsf", 12, True),
            "fred_md": ("fred_md_dataset.tsf", 12, False),
            "traffic_weekly": ("traffic_weekly_dataset.tsf", 8, False),
            "electricity_weekly": ("electricity_weekly_dataset.tsf", 8, True),
            "solar_weekly": ("solar_weekly_dataset.tsf", 5, False),
            "kaggle_web_traffic_weekly": ("kaggle_web_traffic_weekly_dataset.tsf", 8, True),
            "kaggle_web_traffic_daily": ("kaggle_web_traffic_dataset_without_missing_values.tsf", 59, True),
            "us_births": ("us_births_dataset.tsf", 30, True),
            "saugeen_river_flow": ("saugeenday_dataset.tsf", 30, False),
            "sunspot": ("sunspot_dataset_without_missing_values.tsf", 30, True),
            "covid_deaths": ("covid_deaths_dataset.tsf", 30, True),
            "traffic_hourly": ("traffic_hourly_dataset.tsf", 168, False),
            "electricity_hourly": ("electricity_hourly_dataset.tsf", 168, True),
            "solar_10_minutes": ("solar_10_minutes_dataset.tsf", 1008, False),
            "rideshare": ("rideshare_dataset_without_missing_values.tsf", 168, False),
            "temperature_rain": ("temperature_rain_dataset_without_missing_values.tsf", 30, False)
            }
