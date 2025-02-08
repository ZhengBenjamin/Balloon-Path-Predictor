import torch.nn as nn
import torch
import scipy
import numpy as np
from generate_data import GenerateData

class Model(nn.Module):

  def __init__(self):
    super(Model, self).__init__()
    
    a = GenerateData()
    a.get_positions()
    data = a.convert_positions()

    data = torch.from_numpy(data).float()

    self.rnn = nn.RNN(input_size=3,
                      hidden_size=3,
                      num_layers=5,
                      batch_first=True)

if __name__ == "__main__":
  model = Model()