
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from plotting import TrainingTrace

from time import time_ns

def compute_model_size(model):
  weight_mem = sum([a.nbytes for a in model.coefs_]) 
  bias_mem = sum([a.nbytes for a in model.intercepts_])
  total_mem= (weight_mem + bias_mem) / 1024

  weight_count = sum([a.size for a in model.coefs_]) 
  bias_count = sum([a.size for a in model.intercepts_])
  total_params = weight_count + bias_count

  return total_mem, total_params

def model_sklearn( train_x, train_y, test_x, test_y, activation_function, num_layers, num_nodes, epochs ):
  
  hidden_layer_sizes = (num_nodes,) * num_layers
  
  model= MLPClassifier(
    solver='sgd',
    activation='relu',
    batch_size=32,
    learning_rate='constant',
    learning_rate_init= 0.05,
    max_iter=epochs,
    hidden_layer_sizes = hidden_layer_sizes,
    verbose= False
  )

  start_time= time_ns()
  model.fit(train_x, train_y)
  end_time= time_ns()
  print(f'\nTraining time: {(end_time-start_time)/1e6:.1f}ms')

  pred_train_y= model.predict( train_x )
  accuracy_train = accuracy_score(pred_train_y, train_y)
  print(f"Accuracy on train set: {accuracy_train * 100:.3f}%")

  pred_y= model.predict( test_x )
  accuracy_test = accuracy_score(test_y, pred_y)
  print(f"Accuracy on test set: {accuracy_test * 100:.3f}%")

  total_kb, total_params= compute_model_size( model )
  print(f"Total learnable parameters: {total_params}")
  print(f"Estimated memory usage: {total_kb} KB")

  trace= TrainingTrace('sklearn')
  trace.set(None, model.loss_curve_, None)
  return trace
