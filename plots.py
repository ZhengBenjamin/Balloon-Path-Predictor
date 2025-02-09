import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Plots:
  def __init__(self, model, x_test, y_test, scaler_x, scaler_y, device):
    self.model = model
    self.x_test = x_test
    self.y_test = y_test
    self.scaler_x = scaler_x
    self.scaler_y = scaler_y
    self.device = device

  def _unscale_data(self, scaled_input, is_input=True):

    if is_input:
      flat = scaled_input.reshape(-1, 5)
      unscaled = self.scaler_x.inverse_transform(flat)
    else:
      flat = scaled_input.reshape(-1, 3)
      unscaled = self.scaler_y.inverse_transform(flat)
    return unscaled.reshape(scaled_input.shape)

  def _reconstruct_path(self, input_unscaled, offsets_unscaled):
    """Reconstruct the full path from the input and offsets."""

    last_position = input_unscaled[-1, :3] # Last position of input to reconstruct out (offsets)

    reconstructed_path = np.cumsum(offsets_unscaled, axis=0) + last_position # Add offsets
    return reconstructed_path

  def _get_sample_data(self, idx):

    x_sample = self.x_test[idx].numpy()
    input_unscaled = self._unscale_data(x_sample, is_input=True)
    
    y_sample = self.y_test[idx].numpy()
    actual_offsets_unscaled = self._unscale_data(y_sample, is_input=False)
    
    with torch.no_grad(): # Sample offsets
        x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(self.device)
        pred_scaled = self.model(x_tensor).cpu().numpy()[0]
    pred_offsets_unscaled = self._unscale_data(pred_scaled, is_input=False)
    
    actual_path = self._reconstruct_path(input_unscaled, actual_offsets_unscaled)
    pred_path = self._reconstruct_path(input_unscaled, pred_offsets_unscaled)
    
    return input_unscaled, actual_path, pred_path

  def generate_path_plots(self, num_samples=3):
    samples = np.random.choice(len(self.x_test), num_samples, replace=False)
    fig = plt.figure(figsize=(18, 6))
    
    for i, idx in enumerate(samples):
      input_path, actual_path, pred_path = self._get_sample_data(idx)
      
      ax = fig.add_subplot(1, 3, i+1, projection='3d')
      
      # Input
      ax.plot(
        input_path[:, 0], 
        input_path[:, 1], 
        input_path[:, 2], 
        color='gray', 
        linestyle='--', 
        label='Already traveled')
      
      # Actual
      ax.plot(
        actual_path[:, 0], 
        actual_path[:, 1], 
        actual_path[:, 2], 
        color='blue', 
        linewidth=2, 
        label='Actual Path')
      
      # Pred
      ax.plot(
        pred_path[:, 0], 
        pred_path[:, 1], 
        pred_path[:, 2], 
        color='red', 
        linestyle='-.', 
        linewidth=1.5, 
        label='Predicted Path')
      
      ax.set_title(f'Sample {idx+1}')
      ax.set_xlabel('X Coordinate')
      ax.set_ylabel('Y Coordinate')
      ax.set_zlabel('Z Coordinate')
      ax.legend()
    
    plt.tight_layout()
    plt.show()