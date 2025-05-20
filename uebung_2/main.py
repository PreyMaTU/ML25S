from scratch_net.data import train_test_split
from scratch_net.activation import ReLu, Sigmoid, Tanh
from scratch_net.net import Net, Layer
from scratch_net.optimizer import GradientDecent
from scratch_net.loss import MSE, BinaryCrossEntropy
from sklearn.preprocessing import MinMaxScaler
## for pytorch version
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_net.net import train_model, test_model
from pytorch_net.net import Net as PyNet

from data_loader import load_csv_from_zip

import numpy as np

[data_df] = load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
  'breast-cancer-diagnostic.shuf.lrn.csv'
])

# Remove non-feature columns and split columns into inputs and output
x = data_df.drop(columns=['class', 'ID'])  
y = data_df['class']

train_x, train_y, test_x, test_y= train_test_split(x, y, split=0.7, random_state=42)

train_x= train_x.to_numpy()
train_y= train_y.to_numpy()
test_x= test_x.to_numpy()
test_y= test_y.to_numpy()

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

net= Net([
  Layer( 10, ReLu(), input_layer_size= 30),
  Layer( 10, ReLu() ),
  Layer( 1, Sigmoid() )
], loss_function= MSE())

net.train( train_x, train_y, GradientDecent(), epochs= 100, batch_size=32)



pred_train = net.predict ( train_x )
pred_y = net.predict( test_x )

# Evaluation
pred_train_binary = (pred_train > 0.5).astype(int)
accuracy_train = np.mean(pred_train_binary == train_y)
print(f"\nAccuracy on train set: {accuracy_train * 100:.2f}%")
pred_y_binary = (pred_y > 0.5).astype(int)
accuracy = np.mean(pred_y_binary == test_y)
print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")


# Pytorch
##################################################################################
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