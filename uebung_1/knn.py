from reporting import eval_prediction, classifier_header
from dataset_loan import encode_dataset_loan, prepare_numeric_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease, prepare_numeric_dataset_heart_disease
from dataset_dota import encode_dataset_dota
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

header= classifier_header('KNN')

def export_kaggle_results(ids_eval, y_eval, dataset_name):
  eval_results_df = pd.DataFrame()
  eval_results_df['ID'] = ids_eval
  eval_results_df['class'] = y_eval

  # print( eval_results_df )

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/'.join(dataset_name, '.sol.csv'), index=False)


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_k1( x, y, x_eval, ids_eval ):
  header()

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k1_scaled( x, y, x_eval, ids_eval ):
  header()

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  # Scale/Normalize
  scaler = MinMaxScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k5_distance( x, y, x_eval, ids_eval ):
  header()

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k5_distance_scaled( x, y, x_eval, ids_eval ):
  header()

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  # Scale/Normalize
  scaler = RobustScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
  neigh.fit(x_train_scaled, y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval


def dataset_breast_cancer_k5_scaled_crossval( x, y ):
  header()
  
  scaler = RobustScaler()
  x_scaled = scaler.fit_transform(x)
  
  scores = []
  for i in range(1,16):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='manhattan')
    #train model with cv of 5 
    cv_scores = cross_val_score(neigh, x_scaled, y, cv=5)
    # append average accuracy for plotting
    scores.append(np.mean(cv_scores))

  plt.plot(range(1,16), scores)
  plt.title("Accuracy of various k for Breast cancer dataset")
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.savefig("./out/knn_breast_cancer_crossvalidation")


############################################################################################
# Dataset Loan:


def select_k_best(x_train, x_test, y_train, k):
  # Select top k features
  # k=1 is really good, but might be super overfitting
  selector = SelectKBest(score_func=f_classif, k=k)

  x_train_selected= selector.fit_transform(x_train, y_train)
  x_test_selected = selector.transform(x_test)

  # print which features exactly where selected
  #print(x_train.columns[selector.get_support()])

  return x_train_selected, x_test_selected



def dataset_loan_k1_scaled( x, y):
  header()

  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train, x_test = prepare_numeric_dataset_loan(x_train, x_test)

  # select only k "best" features
  x_train, x_test = select_k_best(x_train, x_test, y_train, 15)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval


def dataset_loan_k5_distance_scaled_euclidean( x, y ):
  header()

  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train, x_test = prepare_numeric_dataset_loan(x_train, x_test)

  # select only k "best" features
  x_train, x_test = select_k_best(x_train, x_test, y_train, 15)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance',metric='euclidean')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)

  # TODO x_eval


def dataset_loan_k5_distance_scaled_manhattan( x, y):
  header()

  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train, x_test = prepare_numeric_dataset_loan(x_train, x_test)

  # select only k "best" features
  x_train, x_test = select_k_best(x_train, x_test, y_train, 15)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval

def dataset_loan_k5_distance_scaled_manhattan_crossval( x, y ):
  header()

  x, y = encode_dataset_loan(x,y)

  # ugly workaround, still use whole x for 'x_train'
  x_scaled, x_dummy = prepare_numeric_dataset_loan(x, x)
  # select only k "best" features
  x_scaled, _ = select_k_best(x_scaled, x_dummy, y, 15)

  scores = []
  for i in range(1,16):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='manhattan')
    #train model with cv of 5 
    cv_scores = cross_val_score(neigh, x_scaled, y, cv=5)
    # append average accuracy for plotting
    scores.append(np.mean(cv_scores))

  plt.plot(range(1,16), scores)
  plt.title("Accuracy of various k for Loan dataset")
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.savefig("./out/knn_loan_15_features_crossvalidation")

  

def dataset_loan_k5_distance_scaled_manhattan_one_feature( x, y):
  header()

  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train, x_test = prepare_numeric_dataset_loan(x_train, x_test)

  # select only k "best" features
  x_train, x_test = select_k_best(x_train, x_test, y_train, k=1)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval


def dataset_loan_k5_distance_scaled_manhattan_one_feature_crossval( x, y ):
  header()
  
  x, y = encode_dataset_loan(x,y)

  # ugly workaround, still use whole x for 'x_train'
  x_scaled, x_dummy = prepare_numeric_dataset_loan(x, x)
  # select only k "best" features
  x_scaled, _ = select_k_best(x_scaled, x_dummy, y, k=1)

  scores = []
  for i in range(1,16):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='manhattan')
    #train model with cv of 5 
    cv_scores = cross_val_score(neigh, x_scaled, y, cv=5)
    # append average accuracy for plotting
    scores.append(np.mean(cv_scores))

  plt.plot(range(1,16), scores)
  plt.title("Accuracy of various k for Loan dataset (one feature)")
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.savefig("./out/knn_loan_one_feature_crossvalidation")

############################################################################################
# Dataset Dota:

def dataset_dota_k1( x, y ):
  header()

  x, y= encode_dataset_dota(x, y)

  # scaling not needed, as everything [-1,1]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred)


############################################################################################
# Dataset Heart Disease:


def dataset_heart_disease_k1_scaled( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train_scaled, x_test_scaled= prepare_numeric_dataset_heart_disease(x_train, x_test)

  neigh = KNeighborsClassifier(n_neighbors=1, metric='manhattan', weights='distance')
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred,True)

  # TODO x_eval


def dataset_heart_disease_k5_scaled( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)
  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train_scaled, x_test_scaled= prepare_numeric_dataset_heart_disease(x_train, x_test)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred,True)

  # TODO x_eval

def dataset_heart_disease_scaled_crossval( x, y ):
  header()

  x, y = encode_dataset_heart_disease(x, y)

  x_scaled, _= prepare_numeric_dataset_heart_disease(x, x)

  scores = []
  for i in range(1,16):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='manhattan')
    #train model with cv of 5 
    cv_scores = cross_val_score(neigh, x_scaled, y, cv=5)
    # append average accuracy for plotting
    scores.append(np.mean(cv_scores))

  plt.plot(range(1,16), scores)
  plt.title("Accuracy of various k for Heart Disease dataset")
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.savefig("./out/knn_heart_disease_crossvalidation")
  

  

  


