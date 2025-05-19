from scratch_net.data import train_test_split
from scratch_net.activation import ReLu, Sigmoid, Tanh
from scratch_net.net import Net, Layer
from scratch_net.optimizer import GradientDecent
from scratch_net.loss import MSE

from data_loader import load_csv_from_zip

import numpy as np

[data_df] = load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
  'breast-cancer-diagnostic.shuf.lrn.csv'
])

# Remove non-feature columns and split columns into inputs and output
x = data_df.drop(columns=['class', 'ID'])  
y = data_df['class']

train_x, train_y, test_x, test_y= train_test_split(x, y, split=0.7, random_state=42)

train_x= train_x.to_numpy()
train_y= train_y.to_numpy()
test_x= test_x.to_numpy()
test_y= test_y.to_numpy()

print(train_x)

net= Net([
  Layer( 10, ReLu(), input_layer_size= 30),
  Layer( 10, ReLu() ),
  Layer( 1, ReLu() )
], loss_function= MSE() )

net.train( train_x, train_y, GradientDecent(), epochs= 1000 )

pred_y = net.predict( test_x )

# Evaluation
accuracy = np.mean(pred_y == test_y)
print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")

