import numpy as np

from typing import Self

from .activation import ActivationFunction
from .loss import LossFunction
from .optimizer import Optimizer

class Layer:
  def __init__(self, perceptron_count: int, activation_function: ActivationFunction, input_layer_size: int= 0):
    self.perceptron_count= perceptron_count
    self.shape= None
    self.weights= None
    self.activation_function= activation_function

    # store last values
    self.last_input = None
    self.last_net = None
    self.last_output = None
    self.d_weights = None

    if input_layer_size > 0:
      self.shape= (self.perceptron_count, input_layer_size)
      self.reset()

  def connect_to(self, previous: Self):
    self.shape= (self.perceptron_count, previous.perceptron_count)
    self._reset()

  def reset(self):
    # weights = row vector
    self.weights= np.random.rand( self.shape[0], self.shape[1] ) - 0.5 # weights in range [-0.5, 0.5]
    print( self.shape, self.weights )
    self.weights= np.append( self.weights, np.zeros([self.shape[0],1]), axis= 1) # Add a column of zeros for the bias weights

  # x = batch of several samples where each sample is a column vector output from the previous layer
  def forward(self, x: np.ndarray, keep_output= False):
    x = np.append( x, np.ones([1,x.shape[1]]), axis= 0)
    n= self.weights @ x
    y= self.activation_function.forward( n )

    if keep_output:
      self.last_input= x
      self.last_net= n
      self.last_output= y
    else:
      self.last_input= None
      self.last_net= None
      self.last_output= None
      self.d_weights= None
    
    return y
  
  def backward(self, next_layer: Self, next_delta: np.ndarray ):
    # activation_function.backward is the derivative of the activation function
    d_net = self.activation_function.backward(self.last_output)
    
    delta = (next_layer.weights.T @ next_delta)[:-1, :] * d_net # weights @ dL * d_net
    self.d_weights = delta @ self.last_input.T / self.last_input.shape[1] # average by batch size

    return delta

  # backward of output layer is different since the total Loss is used
  def backward_output_layer(self, loss_function: LossFunction, y_true):
    dL_dout = loss_function.backward(y_true, self.last_output)
    d_net = self.activation_function.backward(self.last_output)
    delta = dL_dout * d_net
    self.d_weights = delta @ self.last_input.T / self.last_input.shape[1] # average by batch size
    return delta


class Net:
  def __init__(self, layers: list[Layer], loss_function: LossFunction):
    self.layers= layers
    self.loss_function= loss_function

    self._build()

  def _build(self):
    # Connect all layers execpt the first one
    prev_layer= None
    for layer in self.layers:
      if prev_layer: 
        layer.connect_to(prev_layer)
      prev_layer= layer

  def _reset(self):
    for layer in self.layers:
      layer.reset()

  def _forward(self, x: np.ndarray, keep_output = False):
    for layer in self.layers:
      x= layer.forward( x, keep_output)

    return x

  def _backward(self, y_true):
    # Start with output layer
    delta = self.layers[-1].backward_output_layer(self.loss_function, y_true)

    # Go backwards through other layers
    for i in reversed(range(len(self.layers) - 1)):
      delta = self.layers[i].backward(delta, self.layers[i + 1].weights)


  def train(self, x, y_true, optimizer: Optimizer, epochs=100, learning_rate= 0.01, batch_size= 32, verbose=True):
    self._reset()

    # transpose data, because the net expects each sample to be a column rather than a row
    x= x.T
    y_true= y_true.T
    
    # optimizer.initialize()

    sample_count = x.shape[1]  # assuming x shape is (input_dim, num_samples)

    for epoch in range(epochs):
      # Shuffle the data
      indices = np.random.permutation(sample_count)
      x_shuffled = x[:, indices]
      y_shuffled = y_true[:, indices]


      for start in range(0, sample_count, batch_size):
        end = start + batch_size
        x_batch = x_shuffled[:, start:end]
        y_batch = y_shuffled[:, start:end]

        self._forward(x_batch, keep_output=True)
        self._backward(y_batch)

        for layer in self.layers:
          layer.weights -= learning_rate * layer.d_weights
      
      if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
        y_pred_epoch = self._forward(x, keep_output=False)
        loss = self.loss_function.forward(y_true, y_pred_epoch)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    

  def predict(self, x):
    # transpose data, because the net expects each sample to be a column rather than a row
    x = x.T
    prediction= self._forward(x)
    return prediction.T





