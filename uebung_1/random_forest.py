from reporting import eval_prediction, classifier_header, append_averaged_cv_scores, store_crossval_scores
from dataset_loan import encode_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease
from dataset_dota import encode_dataset_dota
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
import time

header= classifier_header('RF')

def export_kaggle_results(ids_eval, y_eval):
  eval_results_df= pd.DataFrame()
  eval_results_df['ID']= ids_eval
  eval_results_df['class']= y_eval

  # print( eval_results_df )

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/breast-cancer-diagnostic.sol.csv', index=False)


############################################################################################
# Dataset Breast Cancer:

# def dataset_breast_cancer_crossval_various_estimators( x, y, x_eval, ids_eval ):
#   header()
# 
#   # Create training/test split
#   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
# 
# 
#   # Train the model
#   print('Training...')
#   model = RandomForestClassifier(n_estimators=50, random_state=42)
#   model.fit(x_train, y_train)
# 
#   # Put the test data into the model to see how well it works
#   y_pred = model.predict(x_test)
# 
#   eval_prediction( x_test, y_test, y_pred)
# 
# 
#   # Let the model predict the labels on the evaluation data set from Kaggle
#   y_eval = model.predict(x_eval)
# 
#   export_kaggle_results(ids_eval, y_eval)



def dataset_breast_cancer_crossval_various_depths( x, y, x_eval, ids_eval ):
  header()
  
  scores= []
  depths= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=300, max_depth= i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    depths.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Breast Cancer Depths', depths, scores)


def dataset_breast_cancer_crossval_unscaled_various_estimators( x, y ):
  header()
  
  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Breast Cancer Unscaled', num_estimators, scores)



def dataset_breast_cancer_crossval_various_estimators( x, y ):
  header()
  
  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Breast Cancer Scaled', num_estimators, scores)



############################################################################################
# Dataset Loan:

def dataset_loan_crossval_various_depths( x, y ):
  header()
  
  x, y= encode_dataset_loan( x, y )

  scores= []
  depths= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=300, max_depth= i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    depths.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Loan Depths', depths, scores)


def dataset_loan_crossval_unscaled_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_loan( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Loan Unscaled', num_estimators, scores)



def dataset_loan_crossval_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_loan( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Loan Scaled', num_estimators, scores)


############################################################################################
# Dataset Dota:

def dataset_dota_crossval_various_depths( x, y ):
  header()
  
  x, y= encode_dataset_dota( x, y )

  scores= []
  depths= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=300, max_depth= i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    depths.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Dota Depths', depths, scores)


def dataset_dota_crossval_unscaled_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_dota( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Dota Unscaled', num_estimators, scores)



def dataset_dota_crossval_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_dota( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Dota Scaled', num_estimators, scores)


############################################################################################
# Dataset Heart Disease:


def dataset_heart_disease_crossval_various_depths( x, y ):
  header()
  
  x, y= encode_dataset_heart_disease( x, y )

  scores= []
  depths= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=300, max_depth= i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    depths.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Heart Disease Depths', depths, scores)


def dataset_heart_disease_crossval_unscaled_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_heart_disease( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Heart Disease Unscaled', num_estimators, scores)



def dataset_heart_disease_crossval_various_estimators( x, y ):
  header()
  
  x, y= encode_dataset_heart_disease( x, y )

  scores= []
  num_estimators= []

  for i in range(10, 500, 25):
    pipe = Pipeline([
      ('imputer', SimpleImputer(strategy = 'most_frequent')),
      ('scaler', RobustScaler()),
      ('rf', RandomForestClassifier(n_estimators=i, random_state=42))
    ] )

    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])
  
    num_estimators.append( i )
    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  store_crossval_scores( 'rf', 'Heart Disease Scaled', num_estimators, scores)


def dataset_heart_disease_holdout_with_split( x, y, split_ratio ):
  
  # Create training/test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= split_ratio, stratify=y, random_state=42)
  
  pipe = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('scaler', RobustScaler()),
  ] )

  x_train= pipe.fit_transform( x_train )
  x_test= pipe.transform( x_test )
  
  model = RandomForestClassifier(n_estimators=50, random_state=42)

  start_time= time.time_ns()
  model.fit(x_train, y_train)

  end_time= time.time_ns()
  fit_time= (end_time - start_time) / (10 ** 6)

  y_pred = model.predict(x_test)

  diy_cv_scores= {
    'split_ratio': split_ratio,
    'fit_time': fit_time,
    'test_f1_weighted': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'test_accuracy': accuracy_score(y_test, y_pred)
  }

  return diy_cv_scores

def dataset_heart_disease_holdout( x, y ):
  header()

  x, y= encode_dataset_heart_disease( x, y )

  scores= []

  scores.append( dataset_heart_disease_holdout_with_split(x, y, 0.2) )
  scores.append( dataset_heart_disease_holdout_with_split(x, y, 0.3) )
  scores.append( dataset_heart_disease_holdout_with_split(x, y, 0.4) )

  scores= pd.DataFrame(scores)

  print( scores )

  return scores

