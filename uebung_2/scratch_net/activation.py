import numpy as np

class ActivationFunction:
  def forward( self, x ):
    return 0
  
  def backward( self, y):
    return 0


class ReLu(ActivationFunction):
  def forward(self, x):
    return np.maximum(0.0, x)
  
  def backward(self, y):
    return (y > 0).astype(float)


class Sigmoid(ActivationFunction):
  def forward(self, x):
    return 1.0 / (1 + np.exp(-x))
  
  def backward(self, y):
    return y * (1 - y)


class Tanh(ActivationFunction):
  def forward(self, x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
  
  def backward(self, y):
    t= self.forward( y )
    return 1 - t ** 2


