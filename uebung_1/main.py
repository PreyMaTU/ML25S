from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

from data_loader import load_csv_from_zip
from reporting import count_missing_values

# import k_nearest_neighbors as knn

# Ensure output directory exists
Path("./out").mkdir(parents=True, exist_ok=True)

config= None

rf= None
nn= None
knn= None

def parse_arguments():
  parser= ArgumentParser()
  parser.add_argument('-b', '--breast_cancer', action='store_true')
  parser.add_argument('-l', '--loan', action='store_true')
  parser.add_argument('-d', '--dota', action='store_true')
  parser.add_argument('-e', '--heart_disease', action='store_true')

  parser.add_argument('-n', '--neural_networks', action= 'store_true')
  parser.add_argument('-r', '--random_forests', action= 'store_true')
  parser.add_argument('-k', '--knn', action= 'store_true')

  args= parser.parse_args()

  if not args.knn and not args.random_forests and not args.neural_networks:
    parser.print_help()
    exit(1)

  if not args.breast_cancer and not args.loan and not args.dota and not args.heart_disease:
    parser.print_help()
    exit(1)

  return args


def dataset_breast_cancer():
  # Load the data frames from the zip file
  data_df, eval_df= load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
    'breast-cancer-diagnostic.shuf.lrn.csv',
    'breast-cancer-diagnostic.shuf.tes.csv'
  ])

  # Remove non-feature columns and split columns into inputs and output
  x = data_df.drop(columns=['class', 'ID'])  
  y = data_df['class']

  x_eval = eval_df.drop(columns=['ID'])
  ids_eval= eval_df['ID']

  # TODO: Handle missing values
  # data_df = data_df.dropna()  # or use fillna()

  if config.neural_networks:
    nn.dataset_breast_cancer_version_01( x, y, x_eval, ids_eval )
    nn.dataset_breast_cancer_version_02( x, y, x_eval, ids_eval )
  
  if config.random_forests:
    rf.dataset_breast_cancer_version_01( x, y, x_eval, ids_eval )
    rf.dataset_breast_cancer_version_02( x, y, x_eval, ids_eval )


  if config.knn:
    knn.dataset_breast_cancer_k1( x, y, x_eval, ids_eval )
    knn.dataset_breast_cancer_k1_scaled(x, y, x_eval, ids_eval)
    knn.dataset_breast_cancer_k5_distance(x, y, x_eval, ids_eval)
    knn.dataset_breast_cancer_k5_distance_scaled(x, y, x_eval, ids_eval)
    knn.dataset_breast_cancer_k5_scaled_crossval(x, y)
    


def dataset_loan():
  # Load the data frames from the zip file
  data_df, eval_df= load_csv_from_zip('184-702-tu-ml-2025-s-loan.zip', [
    'loan-10k.lrn.csv',
    'loan-10k.tes.csv'
  ])

  # Remove non-feature columns and split columns into inputs and output
  x = data_df.drop(columns=['grade', 'ID'])  
  y = data_df['grade']

  x_eval = eval_df.drop(columns=['ID'])
  ids_eval= eval_df['ID']
  
  if config.neural_networks:
    nn.dataset_loan_version_01( x, y, x_eval, ids_eval )
  
  if config.random_forests:
    rf.dataset_loan_version_01( x, y, x_eval, ids_eval )
    rf.dataset_loan_version_02( x, y, x_eval, ids_eval )

  if config.knn:
    knn.dataset_loan_k1_scaled(x, y)
    knn.dataset_loan_k5_distance_scaled_euclidean(x, y)
    knn.dataset_loan_k5_distance_scaled_manhattan(x, y)
    knn.dataset_loan_k5_distance_scaled_manhattan_one_feature(x, y)
    knn.dataset_loan_k5_distance_scaled_manhattan_crossval(x, y)
    knn.dataset_loan_k5_distance_scaled_manhattan_one_feature_crossval(x, y)
    
    


def dataset_dota():
  # Load the data frames from the zip file
  [data_df]= load_csv_from_zip('dota2+games+results.zip', ['dota2Train.csv'], False)

  # Make sure all column names are strings
  data_df.columns = data_df.columns.astype(str)

  # Dataset structure: win, clusterid, gamemode, gametype, hero1, ..., heroX
  data_df.rename(columns={
    '0': 'win',
    '1' : 'clusterid',
    '2' : 'gamemode',
    '3': 'gametype'
  }, inplace= True)

  # Separate into features and target
  x = data_df.drop(columns=['win']) 
  y = data_df['win']

  if config.neural_networks:
    nn.dataset_dota_version_01(x, y)

  if config.random_forests:
    rf.dataset_dota_version_01(x, y)

  if config.knn:
    knn.dataset_dota_k1(x, y)
    


def dataset_heart_disease():
  # fetch dataset 
  heart_disease = fetch_ucirepo(id=45) 

    
  x = heart_disease.data.features 
  # Target has 5 classes: 0-4
  y = heart_disease.data.targets 

  # Check the distriubution of the target labels in the dataset
  # 164 55 36 35 13
  #print(np.bincount(y.to_numpy()[:,0]))

  if config.neural_networks:
    nn.dataset_heart_disease_version_01(x, y)

  if config.random_forests:
    y = np.ravel(y)
    rf.dataset_heart_disease_version_01(x, y)

  if config.knn:
    y = np.ravel(y)
    knn.dataset_heart_disease_k1_scaled(x, y)
    knn.dataset_heart_disease_k5_scaled(x, y)
    knn.dataset_heart_disease_scaled_crossval(x, y)


def main():
  global config, rf, nn, knn

  config= parse_arguments()

  if config.neural_networks:
    import neural_networks as neural_networks_module
    nn= neural_networks_module
  
  if config.random_forests:
    import random_forest as random_forest_module
    rf= random_forest_module

  if config.knn:
    import knn as knn_module
    knn= knn_module

  if config.breast_cancer:
    dataset_breast_cancer()
    
  if config.loan:
    dataset_loan()
    
  if config.dota:
    dataset_dota()
    
  if config.heart_disease:
    dataset_heart_disease()

if __name__ == '__main__':
  main()
