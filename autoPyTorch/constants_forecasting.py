# The cosntant values for time series forecasting comes from
# https://github.com/rakshitha123/TSForecasting/blob/master/experiments/deep_learning_experiments.py
# seasonality map, maps a frequency value to a number
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

MAX_WINDOW_SIZE_BASE = 500

