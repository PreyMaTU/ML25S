from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from data_loader import load_csv_from_zip
import neural_networks as nn
import random_forest as rf
import knn
# import k_nearest_neighbors as knn

# Ensure output directory exists
Path("./out").mkdir(parents=True, exist_ok=True)

config= None

def parse_arguments():
  parser= ArgumentParser()
  parser.add_argument('-b', '--breast_cancer', action='store_true')
  parser.add_argument('-l', '--loan', action='store_true')
  parser.add_argument('-d', '--dota', action='store_true')
  parser.add_argument('-e', '--heart_disease', action='store_true')

  parser.add_argument('-n', '--neural_networks', action= 'store_true')
  parser.add_argument('-r', '--random_forests', action= 'store_true')
  parser.add_argument('-k', '--knn', action= 'store_true')

  return parser.parse_args()


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

#  if config.neural_networks:
#    nn.dataset_loan_version_01( x, y, x_eval, ids_eval )
#    nn.dataset_loan_version_02( x, y, x_eval, ids_eval )
  
#  if config.random_forests:
#    rf.dataset_loan_version_01( x, y, x_eval, ids_eval )
#    rf.dataset_loan_version_02( x, y, x_eval, ids_eval )

  if config.knn:
    knn.dataset_loan_version_01( x, y, x_eval, ids_eval )
    knn.dataset_loan_version_02( x, y, x_eval, ids_eval )
    knn.dataset_loan_version_03( x, y, x_eval, ids_eval )


def dataset_dota():
  #TODO:
  pass


def dataset_heart_disease():
  #TODO:
  pass


def main():
  global config

  config= parse_arguments()

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
