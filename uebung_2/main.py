
from data_loader import load_csv_from_zip
from scratch_net.data import train_test_split

from model_scratch_net import *
from model_pytorch import *

import numpy as np

[data_df] = load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
  'breast-cancer-diagnostic.shuf.lrn.csv'
])

# Remove non-feature columns and split columns into inputs and output
x = data_df.drop(columns=['class', 'ID'])  

data_df['class_inv']= ~data_df['class']
y = data_df[['class', 'class_inv']]

train_x, train_y, test_x, test_y= train_test_split(x, y, split=0.7, random_state=42)

train_x= train_x.to_numpy()
train_y= train_y.to_numpy()
test_x= test_x.to_numpy()
test_y= test_y.to_numpy()


# TODO: 1x loss function, grid search, measurement of total number of learnable parameters and virtual RAM
# model_scratch_net( train_x, train_y, test_x, test_y )

model_pytorch( train_x, train_y, test_x, test_y )
