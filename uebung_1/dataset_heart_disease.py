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


