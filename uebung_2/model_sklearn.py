
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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
    verbose= True
  )

  model.fit(train_x, train_y)

  pred_train_y= model.predict( train_x )
  accuracy_train = accuracy_score(pred_train_y, train_y)
  print(f"\nAccuracy on train set: {accuracy_train * 100:.2f}%")

  pred_y= model.predict( test_x )
  accuracy_test = accuracy_score(test_y, pred_y)
  print(f"\nAccuracy on test set: {accuracy_test * 100:.2f}%")
