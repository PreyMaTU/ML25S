import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_net.net import train_model, test_model, get_pytorch_model_stats
from pytorch_net.net import Net as PyNet

from time import time_ns
import numpy as np

def model_pytorch( train_x, train_y, test_x, test_y, activation_function, num_layers, num_nodes, epochs ):

  # Convert dataset to tensors
  x_train_tensor = torch.tensor(train_x, dtype=torch.float32)
  y_train_tensor = torch.tensor(train_y, dtype=torch.float32)

  x_test_tensor = torch.tensor(test_x, dtype=torch.float32)
  y_test_tensor = torch.tensor(test_y, dtype=torch.float32)

  train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32)

  # Initialize model
  input_dims = x_train_tensor.shape[1]
  output_dims = y_train_tensor.shape[1]
  hidden_layers = [num_layers, num_nodes]  

  model = PyNet(input_dims, output_dims, hidden_layers)

  # no need to pass activation fucntion, both dataset-setups use ReLu (from gridsearch)
  start_time= time_ns()
  trace= train_model(model, train_loader, epochs=epochs, learning_rate=0.05)
  end_time= time_ns()
  print(f'\nTraining time: {(end_time-start_time)/1e6:.1f}ms')


  test_model(model, test_loader)

  # print 
  params, vram = get_pytorch_model_stats(model)
  print(f"Total learnable parameters: {params}")
  print(f"Estimated memory usage: {vram:.2f} KB")

  return trace
