from reporting import compare_labels, report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def eval_prediction( x_test, y_test, y_pred ):
  # Report some metrics on the model's quality
  report(y_test, y_pred)

  # Print the true and predicted labels
  #compare_labels(x_test, y_test, y_pred)


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_version_01( x, y, x_eval, ids_eval ):

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)


def dataset_breast_cancer_version_02( x, y, x_eval, ids_eval ):
  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights= 'uniform')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)


def dataset_breast_cancer_version_03( x, y, x_eval, ids_eval ):
  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights= 'distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)

############################################################################################
# Dataset Loan:
#TODO:

############################################################################################
# Dataset Dota:
#TODO:

############################################################################################
# Dataset Heart Disease:
#TODO: