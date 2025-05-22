from scratch_net.activation import ReLu, Sigmoid, Tanh
from scratch_net.net import Net, Layer
from scratch_net.optimizer import GradientDecent
from scratch_net.loss import MSE, CrossEntropyLoss
import numpy as np

import itertools

def grid_search(train_x, train_y, test_x, test_y, epochs):

  input_size = train_x.shape[1]
  output_size = train_y.shape[1]

  activation_functions = [ReLu(), Sigmoid(), Tanh()]
  num_layers_options = [2, 3, 4, 5]
  num_nodes_options = [5, 10, 20, 50]
  
  best_config = None
  best_accuracy = 0

  # itertools.product does the cartesian product for the 3 options (Auskreuzen)
  for activation, num_layers, num_nodes in itertools.product(activation_functions, num_layers_options, num_nodes_options):
    
    # Define the network architecture
    layers = [Layer(num_nodes, activation, input_layer_size=input_size)]
    for _ in range(num_layers - 1):
      layers.append(Layer(num_nodes, activation))
    layers.append(Layer(output_size, Sigmoid()))  # Output layer

    net = Net(layers, loss_function=MSE())
    net.train(train_x, train_y, GradientDecent(), epochs=epochs, batch_size=32, learning_rate=0.05, verbose=False)

    pred_test = net.predict(test_x)
    pred_test_classified = np.argmax(pred_test, axis=1)
    test_truth = np.argmax(test_y, axis=1)
    accuracy = np.mean(pred_test_classified == test_truth)

    activation_name= activation.__class__.__name__
    print(f"Config: Activation={activation_name}, Layers={num_layers}, Nodes={num_nodes} -> Accuracy: {accuracy:.3f}")

    # Track the best configuration
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_config = (activation_name, num_layers, num_nodes)

  print("\nBest Configuration:")
  print(f"Activation Function: {best_config[0]}, Number of Layers: {best_config[1]}, Number of Nodes per Layer: {best_config[2]}")
  print(f"Best Test Accuracy: {best_accuracy:.3f}")





def model_scratch_net( train_x, train_y, test_x, test_y, activation_function, num_layers, num_nodes, epochs):

  # Run the grid search
  # grid_search(train_x, train_y, test_x, test_y, epochs=epochs)
  
  # Breast Cancer dataset: ReLu, 2 layers, 20 Nodes
  # Loan dataset: ReLu, 2 layers, 5 Nodes

  input_size = train_x.shape[1]
  output_size = train_y.shape[1]
  
  # build network
  layers = [Layer(num_nodes, activation_function, input_layer_size=input_size)]
  for _ in range(num_layers - 1):
    layers.append(Layer(num_nodes, activation_function))
  layers.append(Layer(output_size, Sigmoid()))  # Output layer

  net = Net(layers, loss_function=MSE())
  net.train(train_x, train_y, GradientDecent(), epochs=epochs, batch_size=32, learning_rate=0.05)

  pred_train = net.predict ( train_x )
  pred_test = net.predict( test_x )

  # Evaluation

  pred_train_classified= np.argmax( pred_train, axis= 1)
  train_truth= np.argmax( train_y, axis= 1)


  accuracy_train = np.mean(pred_train_classified == train_truth)
  print(f"\nAccuracy on train set: {accuracy_train * 100:.2f}%")

  pred_test_classified= np.argmax( pred_test, axis= 1)
  test_truth= np.argmax( test_y, axis= 1)

  accuracy = np.mean(pred_test_classified == test_truth)
  print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
