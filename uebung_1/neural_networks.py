
from dataset_loan import encode_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease
from dataset_dota import encode_dataset_dota

from reporting import classifier_header, append_averaged_cv_scores, store_crossval_scores
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

header= classifier_header('NN')

def generate_learning_rates():
    return [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]

############################################################################################
# Dataset Breast Cancer:
def dataset_breast_cancer_cv_various_layersizes( x, y, scaler_type="none" ):
  
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

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_breast_cancer_cv_various_learningrates( x, y, scaler_type="none" ):

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
                          learning_rate_init=learning_rate,
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  return learning_rates, scores


def dataset_breast_cancer_cv_various_learningrates_minmax(x, y):
  header()
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate MinMax', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_standard(x, y):
  header()
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'standard')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate Standard', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_robust(x, y):
  header()
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y, 'robust')
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate Robust', learning_rates, scores)

def dataset_breast_cancer_cv_various_learningrates_noscale(x, y):
  header()
  learning_rates, scores = dataset_breast_cancer_cv_various_learningrates(x, y)
  store_crossval_scores( 'NN', 'Breast Cancer Learning Rate No Scale', learning_rates, scores)



def dataset_breast_cancer_cv_various_layersizes_minmax(x, y):
  header()
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes MinMax', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_standard(x, y):
  header()
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'standard')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes Standard', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_robust(x, y):
  header()
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y, 'robust')
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes Robust', x_values, scores)

def dataset_breast_cancer_cv_various_layersizes_noscale(x, y):
  header()
  x_values, scores = dataset_breast_cancer_cv_various_layersizes(x, y)
  store_crossval_scores( 'NN', 'Breast Cancer Layer Sizes No Scale', x_values, scores)



def dataset_breast_cancer_kaggle(x, y, x_eval, ids_eval, validate_before= True):
  header()

  model=  MLPClassifier(
    solver='adam',
    activation='relu',
    learning_rate='constant',
    max_iter=2000,
    hidden_layer_sizes= (128, 64, 32, 16)
  )

  pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', MinMaxScaler()),  
    ('nn', model)
  ] )

  if validate_before:
    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    for metric in ['test_accuracy', 'test_f1_weighted', 'fit_time']:
      print(f"{metric}: {cv_scores[metric].mean():.4f}")

  pipe.fit(x, y)


  # Let the model predict the labels on the evaluation data set from Kaggle
  y_eval = pipe.predict(x_eval)
  
  eval_results_df= pd.DataFrame()
  eval_results_df['ID']= ids_eval
  eval_results_df['class']= y_eval
  eval_results_df['class']= eval_results_df['class'].apply(lambda x: str(x).lower())

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/breast-cancer-diagnostic.sol.ex.csv', index=False)


############################################################################################
# Dataset Loan:

def dataset_loan_cv_various_layersizes( x, y, scaler_type="none" ):
    
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
                          max_iter=300,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_loan_cv_various_learningrates( x, y, scaler_type="none" ):
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
                          learning_rate_init=learning_rate,
                          max_iter=300,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  return learning_rates, scores

def dataset_loan_cv_various_learningrates_minmax(x, y):
  header()
  learning_rates, scores = dataset_loan_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Loan Learning Rate MinMax', learning_rates, scores)

def dataset_loan_cv_various_layersizes_minmax(x, y):
  header()
  x_values, scores = dataset_loan_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Loan Layer Sizes MinMax', x_values, scores)


def dataset_loan_kaggle(x, y, x_eval, ids_eval, validate_before= True):
  header()

  x, y= encode_dataset_loan( x, y )

  model=  MLPClassifier(
    solver='adam',
    activation='relu',
    learning_rate='constant',
    max_iter=2000,
    hidden_layer_sizes= (32, 16)
  )

  pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', MinMaxScaler()),  
    ('nn', model)
  ] )

  if validate_before:
    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    for metric in ['test_accuracy', 'test_f1_weighted', 'fit_time']:
      print(f"{metric}: {cv_scores[metric].mean():.4f}")

  pipe.fit(x, y)

  # Hack: The evaluation set contains some weirdness we have to fix
  x_eval['loan_status']= x_eval['loan_status'].apply(lambda x: None if x.lower() == 'default' else x)
  x_eval['home_ownership']= x_eval['home_ownership'].apply(lambda x: 'OTHER' if x.lower() == 'none' else x)

  # Let the model predict the labels on the evaluation data set from Kaggle
  x_eval, _= encode_dataset_loan( x_eval, None )
  y_eval = pipe.predict(x_eval)
  
  grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

  eval_results_df= pd.DataFrame()
  eval_results_df['ID']= ids_eval
  eval_results_df['grade']= y_eval
  eval_results_df['grade']= eval_results_df['grade'].map( grade_mapping )

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/loan-10k.sol.ex.csv', index=False)



############################################################################################
# Dataset Dota:

def dataset_dota_cv_various_layersizes( x, y, scaler_type="none" ):

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
                          max_iter=30,
                          hidden_layer_sizes= layer
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_dota_cv_various_learningrates( x, y, scaler_type="none" ):

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
                          learning_rate_init=learning_rate,
                          max_iter=30,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  return learning_rates, scores

def dataset_dota_cv_various_learningrates_minmax(x, y):
  header()
  learning_rates, scores = dataset_dota_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Dota Learning Rate MinMax', learning_rates, scores)

def dataset_dota_cv_various_layersizes_minmax(x, y):
  header()
  x_values, scores = dataset_dota_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Dota Layer Sizes MinMax', x_values, scores)



############################################################################################
# Dataset Heart Disease:

############################ Hyperparameter Tweaking ############################
def dataset_heart_disease_cv_various_layersizes( x, y, scaler_type="none" ):
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

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  x_values= [ str(x) for x in hidden_layer_size_array ]
  return x_values, scores


def dataset_heart_disease_cv_various_learningrates( x, y, scaler_type="none" ):
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
                          learning_rate_init=learning_rate,
                          max_iter=2000,
                          hidden_layer_sizes= (64, 32, 16)
                          ))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  return learning_rates, scores


def dataset_heart_disease_cv_various_learningrates_minmax(x, y):
  header()
  learning_rates, scores = dataset_heart_disease_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Learning Rate MinMax', learning_rates, scores)


def dataset_heart_disease_cv_various_layersizes_minmax(x, y):
  header()
  x_values, scores = dataset_heart_disease_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Layer Sizes MinMax', x_values, scores)


def dataset_heart_disease_binary_cv_various_learningrates_minmax(x, y):
  header()
  learning_rates, scores = dataset_heart_disease_cv_various_learningrates(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Binary Learning Rate MinMax', learning_rates, scores)


def dataset_heart_disease_binary_cv_various_layersizes_minmax(x, y):
  header()
  x_values, scores = dataset_heart_disease_cv_various_layersizes(x, y, 'minmax')
  store_crossval_scores( 'NN', 'Heart Disease Binary Layer Sizes MinMax', x_values, scores)
