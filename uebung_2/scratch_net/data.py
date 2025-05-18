import pandas as pd
import numpy as np

# data given as a pandas dataframe
def train_test_split( x, y, split= 0.7, random_state=None ):
  
  if random_state ==  None:
    random_state = np.random.randint()
  x = x.sample(frac=1, random_state= random_state)
  y = y.sample(frac=1, random_state= random_state)

  num_rows = x.shape[0]
  train_size = int(num_rows*split)

  train_x = x[0:train_size]
  test_x = x[train_size:]
  train_y = y[0:train_size]
  test_y = y[train_size:]

  return train_x, train_y, test_x, test_y
