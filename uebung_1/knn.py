from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from reporting import eval_prediction, classifier_header, plot_crossval_scores, append_averaged_cv_scores, store_crossval_scores
from dataset_loan import encode_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease
from dataset_dota import encode_dataset_dota
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

header= classifier_header('KNN')


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_no_scale_cv_various_k( x, y ):
  header()
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Breast Cancer No Scale', None, scores)


def dataset_breast_cancer_standard_cv_various_k( x, y ):
  header()
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', StandardScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Breast Cancer Standard', None, scores)



def dataset_breast_cancer_minmax_cv_various_k( x, y ):
  header()
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Breast Cancer MinMax', None, scores)



############################################################################################
# Dataset Loan:

def dataset_loan_no_scale_cv_various_k( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('selector', SelectKBest(score_func=f_classif, k=10)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Loan No Scale 10 Features', None, scores)

def dataset_loan_standard_cv_various_k( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', StandardScaler()),  
      ('selector', SelectKBest(score_func=f_classif, k=10)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Loan Standard 10 Features', None, scores)


def dataset_loan_minmax_all_features_cv_various_k( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Loan MinMax All Features', None, scores)

def dataset_loan_minmax_cv_various_k( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('selector', SelectKBest(score_func=f_classif, k=10)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Loan MinMax 10 Features', None, scores)



def dataset_loan_minmax_one_feature_cv_various_k( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('selector', SelectKBest(score_func=f_classif, k=1)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Loan MinMax 1 Feature', None, scores)

############################################################################################
# Dataset Dota:

def dataset_dota_no_scale_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_dota(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('selector', SelectKBest(score_func=f_classif, k=30)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance',n_jobs=-1))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  
  store_crossval_scores( 'knn', 'Dota No scale', None, scores)

def dataset_dota_minmax_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_dota(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),
      ('selector', SelectKBest(score_func=f_classif, k=30)),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance',n_jobs=-1))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)
  
  store_crossval_scores( 'knn', 'Dota MinMax', None, scores)


############################################################################################
# Dataset Heart Disease:

def dataset_heart_disease_no_scale_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Heart Disease No Scale', None, scores)

def dataset_heart_disease_standard_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', StandardScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Heart Disease Standard', None, scores)

  

def dataset_heart_disease_minmax_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Heart Disease MinMax', None, scores)

def dataset_heart_disease_minmax_cv_various_k_binary( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', MinMaxScaler()),  
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])

    append_averaged_cv_scores( scores, cv_scores )

  scores= pd.DataFrame(scores)

  store_crossval_scores( 'knn', 'Heart Disease MinMax Binary', None, scores)

  
