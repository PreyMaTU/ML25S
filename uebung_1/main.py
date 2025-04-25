from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import time
from ucimlrepo import fetch_ucirepo 

from data_loader import load_csv_from_zip
from plotting import plotting
from stored_scores import export_stored_crossval_scores, import_stored_crossval_scores

# import k_nearest_neighbors as knn

# Ensure working directories exist
Path("./out").mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)

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
  parser.add_argument('-C', '--compare_models', action= 'store_true')

  parser.add_argument('-p', '--plotting', action='store_true')

  parser.add_argument('--load', nargs='+')
  parser.add_argument('--save')

  args= parser.parse_args()

  if args.compare_models:
    args.knn= True
    args.random_forests= True
    args.neural_networks= True

  if not args.knn and not args.random_forests and not args.neural_networks and not args.load:
    parser.print_help()
    exit(1)

  if not args.breast_cancer and not args.loan and not args.dota and not args.heart_disease and not args.load:
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
    nn.dataset_breast_cancer_cv_various_learningrates_minmax(x, y)
    nn.dataset_breast_cancer_cv_various_learningrates_standard(x, y)
    nn.dataset_breast_cancer_cv_various_learningrates_robust(x, y)
    nn.dataset_breast_cancer_cv_various_learningrates_noscale(x, y)
    
    nn.dataset_breast_cancer_cv_various_layersizes_minmax(x, y)
    nn.dataset_breast_cancer_cv_various_layersizes_standard(x, y)
    nn.dataset_breast_cancer_cv_various_layersizes_robust(x, y)
    nn.dataset_breast_cancer_cv_various_layersizes_noscale(x, y)


  if config.random_forests:
    rf.dataset_breast_cancer_crossval_various_depths( x, y, x_eval, ids_eval )
    rf.dataset_breast_cancer_crossval_unscaled_various_estimators( x, y )
    rf.dataset_breast_cancer_crossval_various_estimators( x, y )


  if config.knn:
    knn.dataset_breast_cancer_no_scale_cv_various_k(x, y)
    knn.dataset_breast_cancer_minmax_cv_various_k(x, y)
    knn.dataset_breast_cancer_standard_cv_various_k(x, y)
    


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
    nn.dataset_loan_cv_various_learningrates_minmax(x, y)
    nn.dataset_loan_cv_various_layersizes_minmax(x, y)
  
  if config.random_forests:
    rf.dataset_loan_crossval_various_depths( x, y )
    rf.dataset_loan_crossval_unscaled_various_estimators( x, y )
    rf.dataset_loan_crossval_various_estimators( x, y )

  if config.knn:
    knn.dataset_loan_minmax_all_features_cv_various_k(x, y)
    knn.dataset_loan_minmax_one_feature_cv_various_k(x, y)
    knn.dataset_loan_minmax_cv_various_k(x, y) 
    knn.dataset_loan_standard_cv_various_k(x, y)  
    knn.dataset_loan_no_scale_cv_various_k(x, y) 
    


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
    nn.dataset_dota_cv_various_learningrates_minmax(x, y)
    nn.dataset_dota_cv_various_layersizes_minmax(x, y)
  
  if config.random_forests:
    rf.dataset_dota_crossval_various_depths( x, y )
    rf.dataset_dota_crossval_unscaled_various_estimators( x, y )
    rf.dataset_dota_crossval_various_estimators( x, y )

  if config.knn:
    knn.dataset_dota_no_scale_cv_various_k(x, y)
    knn.dataset_dota_minmax_cv_various_k(x, y)
    


def dataset_heart_disease():
  # fetch dataset 
  heart_disease = fetch_ucirepo(id=45) 

    
  x = heart_disease.data.features 
  # Target has 5 classes: 0-4
  y = heart_disease.data.targets

  # combine heart disease to single class, essentially binary classification
  y_binary = y.copy()
  y_binary[y_binary > 0] = 1

  # Check the distriubution of the target labels in the dataset
  # 164 55 36 35 13
  #print(np.bincount(y.to_numpy()[:,0]))

  y = np.ravel(y)
  y_binary = np.ravel(y_binary)

  if config.neural_networks:
    nn.dataset_heart_disease_cv_various_learningrates_minmax(x, y)
    nn.dataset_heart_disease_cv_various_layersizes_minmax(x, y)
  
  if config.random_forests:
    rf.dataset_heart_disease_crossval_various_depths( x, y )
    rf.dataset_heart_disease_crossval_unscaled_various_estimators( x, y )
    rf.dataset_heart_disease_crossval_various_estimators( x, y )
    rf.dataset_heart_disease_binary_crossval_various_estimators(x, y_binary)

  if config.knn:
    knn.dataset_heart_disease_no_scale_cv_various_k(x, y)
    knn.dataset_heart_disease_standard_cv_various_k(x, y)
    knn.dataset_heart_disease_minmax_cv_various_k(x, y)
    knn.dataset_heart_disease_minmax_cv_various_k_binary(x, y_binary)



def main():
  global config, rf, nn, knn

  start_time= time.time()

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

  if config.load:
    for file_name in config.load:
      import_stored_crossval_scores( file_name )

  if config.breast_cancer:
    dataset_breast_cancer()
    
  if config.loan:
    dataset_loan()
    
  if config.dota:
    dataset_dota()
    
  if config.heart_disease:
    dataset_heart_disease()

  if config.save:
    export_stored_crossval_scores( config.save )

  if config.plotting:
    plotting()

  run_time= time.time()- start_time
  print(f'\nDone. Script took {run_time}s')

if __name__ == '__main__':
  main()
