import numpy as np

class Optimizer():
  def initialize(self, layers):
    pass

  def update(self, layers, learning_rate):
    pass


class GradientDecent(Optimizer):
  def initialize(self, layers):
    pass
    
  def update(self, layers, learning_rate):
    for layer in layers:
      layer.weights -= learning_rate * layer.d_weights
      