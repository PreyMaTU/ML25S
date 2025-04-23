from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from reporting import eval_prediction, classifier_header
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






############################################################################################
# Dataset Loan:

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



  y_pred = neigh.predict(x_test)



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


############################################################################################
# Dataset Dota:

def dataset_dota_minmax_cv_various_k( x, y ):
  header()

  x, y = encode_dataset_dota(x, y)
  
  scores= []
  for i in range(1,42):
    pipe = Pipeline([
      ('imputer', SimpleImputer()),
      ('scaler', StandardScaler()),
      ('knn', KNeighborsClassifier(n_neighbors=i, weights='distance'))
    ] )

    #train model with cv of 5 
    cv_scores = cross_validate(pipe, x, y, cv=5, scoring=['accuracy','f1_weighted'])


############################################################################################
# Dataset Heart Disease:

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




