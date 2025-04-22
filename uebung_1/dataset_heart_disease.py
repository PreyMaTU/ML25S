from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd


def encode_dataset_heart_disease ( x: pd.DataFrame, y: pd.Series ):

  x = pd.get_dummies(x, columns=['cp', 'restecg', 'slope', 'thal'])

  return x, y


def prepare_numeric_dataset_heart_disease( x ):
  pass
