
from dataset_constants import dataset_loan_numeric_distributed_columns, dataset_loan_numeric_ordinal_columns
from reporting import report, compare_labels
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

import pandas as pd




def eval_prediction( x_test, y_test, y_pred ):
  y_pred= y_pred.round().astype(int)

  # Report some metrics on the model's quality
  report(y_test, y_pred)

  compare_labels(x_test, y_test, y_pred)


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_version_01( x ,y, x_eval, ids_eval ):
  set_random_seed(42)

  # Convert types of columns to numerical values
  y = y.astype(int)  # convert True/False to 1/0

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # Scale/Normalize
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  # Build model
  model = Sequential([
      Dense(12, activation='relu', input_shape=(x_train_scaled.shape[1],)),
      #Dense(64, activation='relu'),
      #Dense(16, activation='relu'),
      #Dense(16, activation='relu'),
      Dense(1, activation='sigmoid')  # binary classification
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Train
  model.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), epochs=20, batch_size=32)

  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test_scaled)

  eval_prediction( x_test, y_test, y_pred )


def dataset_breast_cancer_version_02( x, y, x_eval, ids_eval ):
  pass


############################################################################################
# Dataset Loan:

def encode_dataset_loan( x: pd.DataFrame, y: pd.Series ):
  y= y.to_frame(name='label')
  y['ord']= pd.Categorical( y.label ).codes

  print( y )

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

  x= x.drop([
    'issue_d_month', 'issue_d_year', 'earliest_cr_line_month', 'earliest_cr_line_year',
    'last_pymnt_d_month', 'last_pymnt_d_year', 'last_credit_pull_d_month', 'last_credit_pull_d_year'
  ], axis=1)

  return x, y

def prepare_numeric_dataset_loan( x_train, x_test, y_train, y_test ):
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


  grade_scaler= MinMaxScaler()
  y_train.ord= grade_scaler.fit_transform(y_train.ord)
  y_test.ord= grade_scaler.transform(y_test.ord)

  return x_train, x_test, y_train, y_test

def dataset_loan_version_01( x, y, x_eval, ids_eval ):
  set_random_seed(42)

  x, y= encode_dataset_loan( x, y )

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # Scale + Outlier
  x_train, x_test, y_train, y_test = prepare_numeric_dataset_loan(x_train, x_test)


  # TODO: Make model!

#  # Build model
#  model = Sequential([
#      Dense(12, activation='relu', input_shape=(x_train_scaled.shape[1],)),
#      #Dense(64, activation='relu'),
#      #Dense(16, activation='relu'),
#      #Dense(16, activation='relu'),
#      Dense(1, activation='sigmoid')  # binary classification
#  ])
#
#  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#  # Train
#  model.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), epochs=20, batch_size=32)
#
#  # Put the test data into the model to see how well it works
#  y_pred = model.predict(x_test_scaled)
#
#  eval_prediction( x_test, y_test, y_pred )
  



############################################################################################
# Dataset Dota:
#TODO:

############################################################################################
# Dataset Heart Disease:
#TODO:
