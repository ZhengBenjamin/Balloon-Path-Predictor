from collections import defaultdict
from retry_requests import retry

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import math

import json
import urllib
import re

class GetData:

  def __init__(self):
    self.windborne_url = "http://a.windbornesystems.com/treasure/"
    self.balloon_pos = {}
    self.missing = []

    for i in range(700):
      self.balloon_pos[i] = []

    self.openmeteo_url = "https://api.open-meteo.com/v1/forecast"
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    self.openmeteo = openmeteo_requests.Client(session = retry_session)

    self.get_positions()

  def get_positions(self):

    for i in range(23, -1, -1):
      print(f"Getting data for hour {i}")
      try:
        with urllib.request.urlopen(self.windborne_url + ((("0" + str(i)) if i < 10 else str(i)) + ".json")) as response:
          try: 
            data = re.sub(r"[\n\t\s]", "", response.read().decode("utf-8"))
            if not data.startswith("[["):
              data = "[" + data

            data = json.loads(data)
          except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")

          for balloon_index in range(700):
            line = data[balloon_index]
            self.balloon_pos[balloon_index].append(line)
            for obj in line:
              if math.isnan(obj):
                self.balloon_pos[balloon_index][-1] = [0, 0, 0]

      except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        self.missing.append(23 - i)
        for balloons in self.balloon_pos.keys():
          self.balloon_pos[balloons].append(None)

  def get_weather_data(self, latitude, longitude, elevation, future, past):

    pressure_levels = [1000, 975,	950, 925,	900, 850,	800, 700,	600, 500,	400, 300,	250, 200,	150, 100, 70, 50, 30]
    elevation * 1000
    est_pressure = 101325 * (1 - (9.80665 * elevation) / (288.15 * 287.05))**(9.80665 / (287.05 * (-0.0065)))
    est_pressure = min(pressure_levels, key=lambda x: abs(x - est_pressure))

    params = {
      "latitude": latitude,
      "longitude": longitude,
      "hourly": [f"wind_speed_{est_pressure}hPa", f"wind_direction_{est_pressure}hPa"],
      "wind_speed_unit": "ms",
      "past_hours": past,
      "forecast_days": 1,
      "forecast_hours": future
    }

    responses = self.openmeteo.weather_api(self.openmeteo_url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_wind_speed = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction = hourly.Variables(1).ValuesAsNumpy()

    return hourly_wind_speed, hourly_wind_direction


    

  