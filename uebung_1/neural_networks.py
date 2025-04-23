
from dataset_loan import encode_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease
from dataset_dota import encode_dataset_dota

from reporting import eval_prediction, classifier_header
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

import pandas as pd
import numpy as np

header= classifier_header('NN')

def generate_learning_rates(start=1e-5, stop=0.3, num=20):
    return np.geomspace(start, stop, num=num).round(8).tolist()

############################################################################################
# Dataset Breast Cancer:
def dataset_breast_cancer_cv_various_layersizes( x, y, scaler_type="none" ):
  header()
  
  # Convert types of columns to numerical values
  y = y.astype(int)  # convert True/False to 1/0
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  hidden_layer_size_array = [
    (16,), 
    (32,), 
    (32, 16), 
    (64, 32), 
    (64, 32, 16),
    (128, 64, 32, 16)
  ]
  for layer in hidden_layer_size_array:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_breast_cancer_cv_various_learningrates( x, y, scaler_type="none" ):
  header()

  # Convert types of columns to numerical values
  y = y.astype(int)  # convert True/False to 1/0
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  learning_rates = generate_learning_rates()

  for learning_rate in learning_rates:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])




def dataset_breast_cancer_cv_various_learningrates_minmax(x, y):
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate MinMax', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_standard(x, y):
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'standard')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate Standard', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_robust(x, y):
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'robust')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate Robust', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_noscale(x, y):
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y)
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate No Scale', learning_rates, scores)



def dataset_breast_cancer_cv_various_layersizes_minmax(x, y):
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes MinMax', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_standard(x, y):
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'standard')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes Standard', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_robust(x, y):
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'robust')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes Robust', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_noscale(x, y):
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y)
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes No Scale', x_values, scores)


############################################################################################
# Dataset Loan:

def dataset_loan_cv_various_layersizes( x, y, scaler_type="none" ):
  header()
  
  x, y= encode_dataset_loan( x, y )
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  hidden_layer_size_array = [
    (16,), 
    (32,), 
    (32, 16), 
    (64, 32), 
    (64, 32, 16),
    (128, 64, 32, 16)
  ]
  for layer in hidden_layer_size_array:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_loan_cv_various_learningrates( x, y, scaler_type="none" ):
  header()

  x, y= encode_dataset_loan( x, y )
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  learning_rates = generate_learning_rates()

  for learning_rate in learning_rates:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])



def dataset_loan_cv_various_learningrates_minmax(x, y):
  learning_rates, scores = dataset_loan_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Loan Learning Rate MinMax', learning_rates, scores)

def dataset_loan_cv_various_layersizes_minmax(x, y):
  x_values, scores = dataset_loan_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Loan Layer Sizes MinMax', x_values, scores)


############################################################################################
# Dataset Dota:

def dataset_dota_cv_various_layersizes( x, y, scaler_type="none" ):
  header()
  
  x, y= encode_dataset_dota( x, y )
  x = x.astype(float)
  y = y.astype(float)
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  hidden_layer_size_array = [
    (16,), 
    (32,), 
    (32, 16), 
    (64, 32), 
    (64, 32, 16),
    (128, 64, 32, 16)
  ]
  for layer in hidden_layer_size_array:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_dota_cv_various_learningrates( x, y, scaler_type="none" ):
  header()

  x, y= encode_dataset_dota( x, y )
  x = x.astype(float)
  y = y.astype(float)

  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  learning_rates = generate_learning_rates()

  for learning_rate in learning_rates:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])


  return learning_rates, scores

def dataset_dota_cv_various_learningrates_minmax(x, y):
  learning_rates, scores = dataset_dota_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Dota Learning Rate MinMax', learning_rates, scores)

def dataset_dota_cv_various_layersizes_minmax(x, y):
  x_values, scores = dataset_dota_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Dota Layer Sizes MinMax', x_values, scores)


############################################################################################
# Dataset Heart Disease:

############################ Hyperparameter Tweaking ############################
def dataset_heart_disease_cv_various_layersizes( x, y, scaler_type="none" ):
  header()
  x, y = encode_dataset_heart_disease(x, y)
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  hidden_layer_size_array = [
    (16,), 
    (32,), 
    (32, 16), 
    (64, 32), 
    (64, 32, 16),
    (128, 64, 32, 16)
  ]
  for layer in hidden_layer_size_array:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_heart_disease_cv_various_learningrates( x, y, scaler_type="none" ):
  header()
  x, y = encode_dataset_heart_disease(x, y)
  
  scaler = None
  match scaler_type:
    case "standard":
      scaler = StandardScaler()   
    case "minmax":
      scaler = MinMaxScaler()   
    case "robust":
      scaler = RobustScaler()   
    case _:
      scaler_type = "none"

  scores = []
  
  learning_rates = generate_learning_rates()

  for learning_rate in learning_rates:
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', scaler),  
      ('nn', MLPClassifier(solver='adam',
                          activation='relu',
                          learning_rate='constant',
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  return learning_rates, scores


def dataset_heart_disease_cv_various_learningrates_minmax(x, y):
  learning_rates, scores = dataset_heart_disease_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Learning Rate MinMax', learning_rates, scores)


def dataset_heart_disease_cv_various_layersizes_minmax(x, y):
  x_values, scores = dataset_heart_disease_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Layer Sizes MinMax', x_values, scores)
