
from llm_net.net import NeuralNet

from time import time_ns

def model_llm_generated( train_x, train_y, test_x, test_y, activation_function, num_layers, num_nodes, epochs ):

  if activation_function and activation_function.__class__.__name__ != 'ReLu':
    raise ValueError(f'The LLM net only has support for ReLu activation (provided {activation_function.__class__.__name__})')

  layer_sizes= [ num_nodes for _ in range(num_layers) ]

  nn = NeuralNet(train_x, train_y, layer_sizes=layer_sizes, learning_rate=0.05)

  start_time= time_ns()
  nn.train(train_x, train_y, epochs=epochs)
  end_time= time_ns()
  print(f'\nTraining time: {(end_time-start_time)/1e6:.1f}ms')

  accuracy= nn.evaluate(test_x, test_y)
  print(f"Accuracy on test set: {accuracy * 100:.3f}%")

  stats= nn.get_model_size_info()
  print(f"Total learnable parameters: {stats['total_parameters']}")
  print(f"Estimated memory usage: {stats['memory_kib']:.2f} KB")
