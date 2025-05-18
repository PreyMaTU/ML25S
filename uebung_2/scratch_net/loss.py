import numpy as np

class LossFunction:
  # Loss
  def forward(self, y_true, y_pred):
    return 0

  # Derivative
  def backward(self, y_true, y_pred):
    return 0

# Mean Squared Error
class MSE(LossFunction):
  def forward(self, y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
  
  def backward(self, y_true, y_pred):
    output_count= np.shape(y_pred)[1]
    return 2.0 / output_count * (y_pred - y_true)


# CCE
class CategoricalCrossEntropy(LossFunction):
  def forward(self, y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon) # Clip so no log(0) 
    return - (np.sum(y_true * np.log(y_pred), axis=1))

  # TODO: backward derivative
