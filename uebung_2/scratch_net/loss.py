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
    return np.mean((y_pred - y_true) ** 2)
  
  def backward(self, y_true, y_pred):
    return 2.0 * (y_pred - y_true)


# TODO: Check correctness of calculation for BCE and CCE

class BinaryCrossEntropy(LossFunction):
    def forward(self, y_pred, y_true):
        # Clip to prevent log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Derivative of BCE w.r.t. y_pred
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

# CCE
class CategoricalCrossEntropy(LossFunction):
  def forward(self, y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon) # Clip so no log(0) 
    return - (np.sum(y_true * np.log(y_pred), axis=1))

  # TODO: backward derivative
