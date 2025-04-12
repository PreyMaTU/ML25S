from data_loader import load_csv_from_zip

train_df, test_df= load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
  'breast-cancer-diagnostic.shuf.lrn.csv',
  'breast-cancer-diagnostic.shuf.tes.csv'
])



# print( train_df )

print('Missing values in the training set:\n', train_df.isnull().sum())
print('Missing values in the testing set:\n', test_df.isnull().sum())
