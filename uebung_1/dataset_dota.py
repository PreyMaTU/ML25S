

from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd



def encode_dataset_dota( x: pd.DataFrame, y: pd.Series ):
  
  x= pd.get_dummies(x, columns=['clusterid', 'gamemode','gametype'])

  # Convert range [-1, 1] -> [0, 1]
  y = (y + 1) // 2

  return x, y



# def prepare_numeric_dataset_dota( x_train, x_test ):
#   return x_train, x_test
