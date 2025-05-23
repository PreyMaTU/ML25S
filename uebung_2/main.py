
from data_loader import load_csv_from_zip
from dataset_loan import encode_dataset_loan
from scratch_net.data import train_test_split
from scratch_net.activation import ReLu
from plotting import plot_loss, plot_accuracy

from model_scratch_net import *
from model_pytorch import *
from model_sklearn import *
from model_llm_generated import *

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import sys, os, pathlib
sys.path.insert(0, os.path.abspath('..'))

pathlib.Path('./out').mkdir(parents=True, exist_ok=True)

# Breast Cancer Dataset
##################################################################################################
if True:
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

  # Scale the input data
  scaler = StandardScaler()
  train_x = scaler.fit_transform(train_x)
  test_x = scaler.transform(test_x)
  
  epochs = 200
  activation_function = ReLu()
  num_layers = 2
  num_nodes = 20

  traces= []

  print("==================================================================")
  print("Train ScratchNet Breast Cancer:")
  traces.append( model_scratch_net( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train Pytorch Breast Cancer:")
  traces.append( model_pytorch( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train Sklearn Breast Cancer:")
  traces.append( model_sklearn( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train LLM-Generated Breast Cancer:")
  traces.append( model_llm_generated( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  plot_loss('Breast Cancer - Training Loss', traces)
  plot_loss('Breast Cancer - Training Loss (DIY)', traces, ['ScratchNet', 'LLM'])
  plot_accuracy('Breast Cancer - Training Accuracy', traces, ['ScratchNet', 'LLM'])

# Loan Dataset
################################################################################################
if True:
  [loan_df] = load_csv_from_zip('184-702-tu-ml-2025-s-loan.zip', [
    'loan-10k.lrn.csv'
  ])


  # Remove non-feature columns and split columns into inputs and output
  loan_x = loan_df.drop(columns=['grade', 'ID'])  

  # One hot encode the labels while keeping the original column
  loan_y = loan_df[['grade']]

  # Encode the dataset (the same as for uebung_1)
  loan_x , loan_y = encode_dataset_loan(loan_x, loan_y)

  # Train/Test split
  train_x, train_y, test_x, test_y= train_test_split(loan_x, loan_y, split=0.7, random_state=42)

  # Convert to numpy arrays
  train_x= train_x.to_numpy()
  train_y_label= train_y['grade'].to_numpy()
  train_y_one_hot= train_y.drop(['grade'], axis=1).to_numpy()
  
  test_x= test_x.to_numpy()
  test_y_label= test_y['grade'].to_numpy()
  test_y_one_hot= test_y.drop(['grade'], axis=1).to_numpy()

  # Scale the input data
  scaler = StandardScaler()
  train_x = scaler.fit_transform(train_x)
  test_x = scaler.transform(test_x)

  epochs = 200
  activation_function = ReLu()
  num_layers = 2
  num_nodes = 5

  traces= []

  print("==================================================================")
  print("Train ScratchNet Loan:")
  traces.append( model_scratch_net( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train Pytorch Loan:")
  traces.append( model_pytorch( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train Sklearn Loan:")
  traces.append( model_sklearn( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  print("==================================================================")
  print("Train LLM-Generated Loan:")
  traces.append( model_llm_generated( train_x, train_y_one_hot, test_x, test_y_one_hot, activation_function, num_layers, num_nodes, epochs=epochs ) )

  plot_loss('Loan - Training Loss', traces)
  plot_loss('Loan - Training Loss (no PyTorch)', traces, ['ScratchNet', 'LLM', 'sklearn'])
  plot_accuracy('Loan - Training Accuracy', traces, ['ScratchNet', 'LLM'])
