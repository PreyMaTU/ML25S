from reporting import compare_labels, report
from dataset_constants import dataset_loan_numeric_distributed_columns, dataset_loan_numeric_ordinal_columns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd


def eval_prediction( x_test, y_test, y_pred, multiclass= False ):
  # Report some metrics on the model's quality
  report(y_test, y_pred,multiclass)

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
  scaler = RobustScaler()
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
  scaler = RobustScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  neigh = KNeighborsClassifier(n_neighbors=5, weights= 'distance')
  neigh.fit(x_train_scaled,y_train)
  y_pred = neigh.predict(x_test_scaled)

  eval_prediction(x_test, y_test, y_pred)

  # TODO x_eval

############################################################################################
# Dataset Loan:

def encode_dataset_loan( x: pd.DataFrame, y: pd.Series ):
  y= y.to_frame(name='label')
  y['ord']= pd.Categorical( y.label ).codes # convert classes A-E to 0-4

  x= x.copy()
  
  # Binary categories
  x['term_60months'] = x.term.map( lambda t : 1 if t.strip() == '60 months' else 0 )
  x['pymnt_plan'] = x.term.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['hardship_flag'] = x.term.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['debt_settlement_flag'] = x.term.map( lambda t : 1 if t.strip().lower() == 'y' else 0 )
  x['initial_list_status_whole'] = x.term.map( lambda t : 1 if t.strip().lower() == 'w' else 0 )
  x['application_type_individual'] = x.term.map( lambda t : 1 if t.strip() == 'Individual' else 0 )
  x['disbursement_method_cash'] = x.term.map( lambda t : 1 if t.strip() == 'Cash' else 0 )

  # Drop all the binary columns that were renamed
  x = x.drop([
    'term', 'application_type', 'disbursement_method', 'initial_list_status'
  ], axis=1)

  # Ordinal categories
  emp_length_mapping = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
  }
  x.emp_length = x.emp_length.map(emp_length_mapping).astype(int)

  # One-Hot encoded
  x = pd.get_dummies(x, columns=[
    'home_ownership', 'loan_status', 'verification_status', 'purpose', 'addr_state'
  ])
  
  # Combine _year + _month columns into a single float
  x['issue_d'] = x.issue_d_year + ( x.issue_d_month / 12.0 )
  x['earliest_cr_line'] = x.earliest_cr_line_year + ( x.earliest_cr_line_month / 12.0 )
  x['last_pymnt_d'] = x.last_pymnt_d_year + ( x.last_pymnt_d_month / 12.0 )
  x['last_credit_pull_d'] = x.last_credit_pull_d_year + ( x.last_credit_pull_d_month / 12.0 )

  # drop old, uncombined columns
  x= x.drop([
    'issue_d_month', 'issue_d_year', 'earliest_cr_line_month', 'earliest_cr_line_year',
    'last_pymnt_d_month', 'last_pymnt_d_year', 'last_credit_pull_d_month', 'last_credit_pull_d_year'
  ], axis=1)

  # only the ordinal encoding is needed
  return x, y['ord']

def scale_dataset_loan ( x_train, x_test ):
  numeric_robust_scaled_columns= dataset_loan_numeric_distributed_columns()
  numeric_minmax_scaled_columns= dataset_loan_numeric_ordinal_columns()

   # Robust Scaling to handle outliers.
  # Why robust scaling and not deleting outliers or capping the value?
  # As we think that the loan data should not contain errors, the outliers should also not be errors in the data
  # Hence we do not want to distort the outliers (by deleting or capping) for the training but just want to make sure
  # that they do not influence the scaling of the data.
  robust_scaler= RobustScaler()
  x_train[numeric_robust_scaled_columns] = robust_scaler.fit_transform(x_train[numeric_robust_scaled_columns])
  x_test[numeric_robust_scaled_columns] = robust_scaler.transform(x_test[numeric_robust_scaled_columns])

  minmax_scaler= MinMaxScaler()
  x_train[numeric_minmax_scaled_columns] = minmax_scaler.fit_transform(x_train[numeric_minmax_scaled_columns])
  x_test[numeric_minmax_scaled_columns] = minmax_scaler.transform(x_test[numeric_minmax_scaled_columns])

  return x_train, x_test

def dataset_loan_k1( x, y, x_eval, ids_eval ):
  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval


def dataset_loan_k1_scaled( x, y, x_eval, ids_eval ):
  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  x_train, x_test = scale_dataset_loan(x_train, x_test)

  neigh = KNeighborsClassifier(n_neighbors=1)
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval


def dataset_loan_k5_distance( x, y, x_eval, ids_eval ):
  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)

  # TODO x_eval


def dataset_loan_k5_distance_scaled( x, y, x_eval, ids_eval ):
  x, y = encode_dataset_loan(x,y)

  # Create training/test split
  # training in the sense that these are used for knn classification
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  x_train, x_test = scale_dataset_loan(x_train, x_test)

  neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
  neigh.fit(x_train,y_train)
  y_pred = neigh.predict(x_test)

  eval_prediction(x_test, y_test, y_pred,multiclass= True)
  # TODO x_eval

############################################################################################
# Dataset Dota:
#TODO:

############################################################################################
# Dataset Heart Disease:
#TODO: