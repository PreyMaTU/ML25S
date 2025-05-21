from scratch_net.activation import ReLu, Sigmoid, Tanh
from scratch_net.net import Net, Layer
from scratch_net.optimizer import GradientDecent
from scratch_net.loss import MSE, BinaryCrossEntropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


  scaler = StandardScaler()
  train_x = scaler.fit_transform(train_x)
  test_x = scaler.transform(test_x)
def model_scratch_net( train_x, train_y, test_x, test_y, epochs):

  net= Net([
    Layer( 10, ReLu(), input_layer_size= 30),
    Layer( 10, ReLu() ),
    Layer( 2, Sigmoid() )
  ], loss_function= MSE())

  net.train(train_x, train_y, GradientDecent(), epochs=epochs, batch_size=32)

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
