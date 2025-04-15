from reporting import compare_labels, report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


def eval_prediction( x_test, y_test, y_pred ):
  # Report some metrics on the model's quality
  report(y_test, y_pred)

  # Print the true and predicted labels
  #compare_labels(x_test, y_test, y_pred)

def export_kaggle_results(ids_eval, y_eval, dataset_name):
  eval_results_df= pd.DataFrame()
  eval_results_df['ID']= ids_eval
  eval_results_df['class']= y_eval

  # print( eval_results_df )

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/'.join(dataset_name,'.sol.csv'), index=False)


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_k1( x, y, x_eval, ids_eval ):

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval

def dataset_breast_cancer_k1_scaled( x, y, x_eval, ids_eval ):

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # Scale/Normalize
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k5_distance( x, y, x_eval, ids_eval ):
  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights= 'distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k5_distance_scaled( x, y, x_eval, ids_eval ):
  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # Scale/Normalize
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  neigh = KNeighborsClassifier(n_neighbors=5, weights= 'distance')
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval

############################################################################################
# Dataset Loan:
#TODO:

############################################################################################
# Dataset Dota:
#TODO:

############################################################################################
# Dataset Heart Disease:
#TODO: