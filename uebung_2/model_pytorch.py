import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_net.net import train_model, test_model, get_pytorch_model_stats
from pytorch_net.net import Net as PyNet

def model_pytorch( train_x, train_y, test_x, test_y ):

  # Convert dataset to tensors
  x_train_tensor = torch.tensor(train_x, dtype=torch.float32)
  y_train_tensor = torch.tensor(train_y, dtype=torch.long)

  x_test_tensor = torch.tensor(test_x, dtype=torch.float32)
  y_test_tensor = torch.tensor(test_y, dtype=torch.long)

  train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
  test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32)

  # Initialize model
  input_dims = x_train_tensor.shape[1]  
  output_dims = 2  # binary classification
  hidden_layers = [10, 10]  

  model = PyNet(input_dims, output_dims, hidden_layers)

  train_model(model, train_loader, epochs=400, learning_rate=0.01)
  test_model(model, test_loader)

  # print 
  params, vram = get_pytorch_model_stats(model)
  print(f"Total learnable parameters: {params}")
  print(f"Estimated VRAM usage: {vram:.2f} KB")
