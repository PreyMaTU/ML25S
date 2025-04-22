from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd


def dataset_heart_disease_numeric_distributed_columns():
  return [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak',
    'ca'
  ]


def dataset_heart_disease_categorical_one_hot_columns():
  return ['cp', 'restecg', 'slope', 'thal']

def dataset_heart_disease_categorical_binary_columns():
  return ['sex', 'fbs', 'exang']


def encode_dataset_heart_disease ( x: pd.DataFrame, y: pd.Series ):

  # When there is a missing value for a column that gets one-hot-encoded
  # then the resulting columns will all be "FALSE". As we have to decide on 
  # some kind of value for a categorical one when it is missing, we decide to
  # simple have false in all resulting columns instead of picking most-frequent one

  one_hot_columns= dataset_heart_disease_categorical_one_hot_columns()
  x = pd.get_dummies(x, columns=one_hot_columns)

  return x, y


def prepare_numeric_dataset_heart_disease( x_train, x_test, imputer_strategy= 'median', scale_values= True ):

  numeric_robust_scaled_columns= dataset_heart_disease_numeric_distributed_columns()
  categorical_binary_columns= dataset_heart_disease_categorical_binary_columns()

  numeric_imputer = SimpleImputer(strategy=imputer_strategy)
  x_train[numeric_robust_scaled_columns] = numeric_imputer.fit_transform(x_train[numeric_robust_scaled_columns])
  x_test[numeric_robust_scaled_columns] = numeric_imputer.transform(x_test[numeric_robust_scaled_columns])

  # No need to impute the one-hot-encoded features, as 'get_dummies' has already replaced them with False
  categorical_imputer = SimpleImputer(strategy='most_frequent')
  x_train[categorical_binary_columns] = categorical_imputer.fit_transform(x_train[categorical_binary_columns])
  x_test[categorical_binary_columns] = categorical_imputer.transform(x_test[categorical_binary_columns])

  if scale_values:
    robust_scaler= RobustScaler()
    x_train[numeric_robust_scaled_columns] = robust_scaler.fit_transform(x_train[numeric_robust_scaled_columns])
    x_test[numeric_robust_scaled_columns] = robust_scaler.transform(x_test[numeric_robust_scaled_columns])

  return x_train, x_test
