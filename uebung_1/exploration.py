
import pandas as pd

def find_loan_dataset_outliers( numeric_cols, x_train ):
  

  lqs= x_train[numeric_cols].quantile(0.01)
  uqs= x_train[numeric_cols].quantile(0.99)

  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print( lqs )
    print( uqs )

    print( ((x_train[numeric_cols] < lqs) | (x_train[numeric_cols] > uqs)).sum() )
