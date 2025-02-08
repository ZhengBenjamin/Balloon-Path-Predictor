from collections import defaultdict
from get_data import GetData

import numpy as np

class GenerateData:

  def __init__(self):
    self.loader = GetData()

    self.balloon_pos = self.loader.balloon_pos
    self.missing = self.loader.missing

  def interpolate_missing(self):
    """Interpolates missing balloon positions using linear interpolation."""

    for balloon_index in self.balloon_pos.keys():
      for missing_hour in self.missing:

        prev_hour, next_hour = None, None

        # Next Known
        for h in range(missing_hour - 1, -1, -1):
          if self.balloon_pos[balloon_index][h] is not None:
            prev_hour = h
            break

        # Prev known known
        for h in range(missing_hour + 1, 24):
          if h < len(self.balloon_pos[balloon_index]) and self.balloon_pos[balloon_index][h] is not None:
            next_hour = h
            break

        # Linear interpolation
        if prev_hour is not None and next_hour is not None:
          prev_value = np.array(self.balloon_pos[balloon_index][prev_hour])
          next_value = np.array(self.balloon_pos[balloon_index][next_hour])
          fraction = (missing_hour - prev_hour) / (next_hour - prev_hour)
          interpolated_value = prev_value + fraction * (next_value - prev_value)
          self.balloon_pos[balloon_index][missing_hour] = interpolated_value.tolist()

        elif prev_hour is not None:
          self.balloon_pos[balloon_index][missing_hour] = self.balloon_pos[balloon_index][prev_hour]
          print(missing_hour)

        elif next_hour is not None:
          self.balloon_pos[balloon_index][missing_hour] = self.balloon_pos[balloon_index][next_hour]

  def convert_positions(self) -> np.ndarray:
    """Converts positions to numpy ndarray"""

    next_pos = np.zeros((len(self.balloon_pos[0]), len(self.balloon_pos), 3))

    for balloon_index, balloon in enumerate(self.balloon_pos):
      
      for hour_index, hour_pos in enumerate(self.balloon_pos[balloon]):
        next_pos[hour_index, balloon_index, 0] = hour_pos[0] # X
        next_pos[hour_index, balloon_index, 1] = hour_pos[1] # Y
        next_pos[hour_index, balloon_index, 2] = hour_pos[2] # Z

    return next_pos

  def gen_vectors(self):
    """
    Generates input and output training vectors for the model
    Input vector = 12 hours, 5 inputs (x, y, z, wind speed, wind dir)
    Output vector = 12 hours, 3 outputs (x, y, z)
    """

    input_cutoff = (len(self.balloon_pos[0]) / 2) + (len(self.balloon_pos[0]) / 4)

    input = np.zeros((input_cutoff, len(self.balloon_pos), 5))
    output = np.zeros((len(self.balloon_pos) - input_cutoff, len(self.balloon_pos), 3))

    # Generate input vectors
    for balloon_index, balloon in enumerate(self.balloon_pos):
      prev_coords = [self.balloon_pos[balloon_index][0], self.balloon_pos[balloon_index][1], self.balloon_pos[balloon_index][2]] # Prev X, Y, Z
      wind_data = self.loader.get_weather_data(prev_coords[0], prev_coords[1], prev_coords[2], 0, 12)
      wind_speed = wind_data[0]
      wind_dir = wind_data[1]
      
      for hour_index in range(input_cutoff):
        data_point = self.balloon_pos[balloon][hour_index]
        input[hour_index, balloon_index, 0] = data_point[0] # X
        input[hour_index, balloon_index, 1] = data_point[1] # Y
        input[hour_index, balloon_index, 2] = data_point[2] # Z

        if abs(data_point[0] - prev_coords[0]) > 0.5 or abs(data_point[1] - prev_coords[1]) > 0.5 or abs(data_point[2] - prev_coords[2]) > 2.5:
          wind_data = self.loader.get_weather_data(data_point[0], data_point[1], data_point[2], 0, 12)
          wind_speed = wind_data[0]
          wind_dir = wind_data[1]

          input[hour_index, balloon_index, 3] = wind_speed[hour_index]
          input[hour_index, balloon_index, 4] = wind_dir[hour_index]
        else:
          input[hour_index, balloon_index, 3] = wind_speed[hour_index]
          input[hour_index, balloon_index, 4] = wind_dir[hour_index]

    # Generate output vectors
    for balloon_index, balloon in enumerate(self.balloon_pos):
      
    

if __name__ == "__main__":
  get_position = GenerateData()
  get_position.interpolate_missing()
  get_position.convert_positions()
  
  data = GetData()
  print(data.get_weather_data(-148.9, -24.8, 11*1000, 0, 12))