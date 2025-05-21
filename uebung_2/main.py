
from data_loader import load_csv_from_zip
from scratch_net.data import train_test_split

from model_scratch_net import *
from model_pytorch import *

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# Pytorch MSE Loss
# LLM version, Sklearn version
# Grid search for scratch_net (Use best version also for other Nets)
# Plots

# Breast Cancer Dataset
##################################################################################################
if False:
  [breast_cancer_df] = load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
    'breast-cancer-diagnostic.shuf.lrn.csv'
  ])

  # Remove non-feature columns and split columns into inputs and output
  breast_cancer_x = breast_cancer_df.drop(columns=['class', 'ID'])  

  # One hot encode the labels while keeping the original column
  breast_cancer_y = breast_cancer_df[['class']]
  breast_cancer_y['class_to_encode']= breast_cancer_y['class']
  breast_cancer_y = pd.get_dummies(breast_cancer_y, columns=['class_to_encode'])

  # Train/Test split
  train_x, train_y, test_x, test_y= train_test_split(breast_cancer_x, breast_cancer_y, split=0.7, random_state=42)

  # Convert to numpy arrays
  train_x= train_x.to_numpy()
  train_y_label= train_y['class'].to_numpy()
  train_y_one_hot= train_y.drop(['class'], axis=1).to_numpy()
  
  test_x= test_x.to_numpy()
  test_y_label= test_y['class'].to_numpy()
  test_y_one_hot= test_y.drop(['class'], axis=1).to_numpy()


  # TODO: grid search, measurement of total number of learnable parameters and virtual RAM
  model_scratch_net( train_x, train_y_one_hot, test_x, test_y_one_hot, epochs=500 )

  # TODO: Use same setup as for scratch net to make compareable
  #       e.g. Activation- and Loss-Function, Optimizer, MLP-Structure, ...
  model_pytorch( train_x, train_y_label, test_x, test_y_label, epochs=500 )


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
