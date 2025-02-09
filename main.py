from model import Model
from generate_data import GenerateData
from plots import Plots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn as nn

scaler_x = None
scaler_y = None

def load_and_preprocess_data():
  """Load data and preprocess it for training."""

  data = GenerateData()
  data.interpolate_missing()
  x, y = data.gen_vectors()  

  x = np.transpose(x, (1, 0, 2))  # (batch, seq_len, 5)
  y = np.transpose(y, (1, 0, 2))  # (batch, seq_len, 3)

  batch_size, seq_len, in_features = x.shape
  _, _, out_features = y.shape

  x_flat = x.reshape(-1, in_features)
  y_flat = y.reshape(-1, out_features)

  scaler_x = MinMaxScaler()
  scaler_y = MinMaxScaler()
  x_flat_scaled = scaler_x.fit_transform(x_flat)
  y_flat_scaled = scaler_y.fit_transform(y_flat)

  x_scaled = x_flat_scaled.reshape(batch_size, seq_len, in_features)
  y_scaled = y_flat_scaled.reshape(batch_size, seq_len, out_features)
  
  x_train, x_test, y_train, y_test = train_test_split(
      x_scaled, y_scaled, test_size=0.2, random_state=42)
  
  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)
  x_test = torch.tensor(x_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)
  
  return x_train, y_train, x_test, y_test, scaler_x, scaler_y


def train_model(model, train_loader, test_loader, epochs, lr, device):
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
  model.train()
  for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)
      
      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item() * batch_x.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    
    # Test set eval 
    if epoch % 100 == 0:
      model.eval()
      test_loss = 0.0
      with torch.no_grad():
        for test_x, test_y in test_loader:
          test_x = test_x.to(device)
          test_y = test_y.to(device)
          test_out = model(test_x)
          loss_test = criterion(test_out, test_y)
          test_loss += loss_test.item() * test_x.size(0)
      test_loss /= len(test_loader.dataset)
      print(f"Epoch {epoch:04d}, Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}")
      model.train()

  torch.save(model.state_dict(), 'model.pth')

def evaluate_unscaled(model, input, scaler_y, device):
  """
  Scaler to convert the model output back to the original scale.
  """
  model.eval()
  with torch.no_grad():
    input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
    scaled_output = model(input_tensor)  # shape: (1, seq_len, 3)
    
    scaled_output_np = scaled_output.cpu().numpy().reshape(-1, 3)
    unscaled_output_np = scaler_y.inverse_transform(scaled_output_np)
    unscaled_output = unscaled_output_np.reshape(scaled_output.shape)
  
  return unscaled_output

def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Process:
  x_train, y_train, x_test, y_test, scaler_x, scaler_y = load_and_preprocess_data()
  
  train_dataset = TensorDataset(x_train, y_train)
  test_dataset = TensorDataset(x_test, y_test)
  batch_size = 64
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
  model = Model(
    input_size=5, 
    hidden_size=25, 
    num_layers=2,
    dropout=0.1,
    output_size=3).to(device)
  
  # Train:
  epochs = 3000
  lr = 0.0005
  train_model(model, train_loader, test_loader, epochs, lr, device) # Comment out if use pre-trained model

  model.load_state_dict(torch.load('model.pth'))
  
  # Eval:
  sample_input = x_test[0].cpu().numpy()  # shape: (seq_len, 5)
  unscaled_prediction = evaluate_unscaled(model, sample_input, scaler_y, device)

  print(f"Pred: {unscaled_prediction[0][:5]}")
  
  y_sample = y_test[0].cpu().numpy().reshape(-1, 3)
  actual = scaler_y.inverse_transform(y_sample).reshape(y_test[0].shape)
  print(f"Sampled: {actual[:5]}")


  sample_indices = [0, 1, 2]
  predictions = []
  actual_vals = []
  for i in sample_indices:
      # Call model
      sample_input = x_test[i].cpu().numpy()  # shape: (seq_len, 5)
      prediction = evaluate_unscaled(model, sample_input, scaler_y, device)  # shape: (seq_len, 3)
      predictions.append(prediction)

      # Actual vals
      y_sample = y_test[i].cpu().numpy().reshape(-1, 3)
      actual = scaler_y.inverse_transform(y_sample).reshape(y_test[i].shape)
      actual_vals.append(actual)

  plotter = Plots(model, x_test, y_test, scaler_x, scaler_y, device)
  plotter.generate_path_plots(num_samples=3)

if __name__ == "__main__":
  main()