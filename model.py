import torch.nn as nn

class Model(nn.Module):
  
  def __init__(self, input_size=5, hidden_size=16, num_layers=2, dropout=0.1, output_size=3):
    """
    Simple LSTM network, sequence -> offsets (x, y, z)
    """
    super(Model, self).__init__()
    
    self.lstm = nn.LSTM(
      input_size, 
      hidden_size, 
      num_layers=num_layers,
      batch_first=True, 
      dropout=dropout)
    
    self.fc = nn.Linear(hidden_size, output_size)
    
  def forward(self, x):
    # x shape: (batch, seq_len, input_size)
    lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
    out = self.fc(lstm_out)     # out: (batch, seq_len, output_size)
    return out