import sklearn.metrics as metrics
import pandas as pd
import sys

def eval_prediction( x_test, y_test, y_pred, multiclass= False ):
  # Report some metrics on the model's quality
  report(y_test, y_pred, multiclass)
  compare_labels(x_test, y_test, y_pred)


def report(y_test, y_pred, multiclass= False):
  # Report some metrics on the model's quality
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  # Depending if the class is binary or multiclass
  if multiclass:
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average='weighted'))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted', zero_division=1.0))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
  else:
    print("Precision:", metrics.precision_score(y_test, y_pred, zero_division=1.0))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))

  # Full report
  print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred, zero_division=1.0))

def compare_labels(x_test, y_test, y_pred):
  results_df = pd.DataFrame( x_test.copy() )
  results_df['True_Label'] = y_test

  results_df.reset_index(drop=True, inplace=True)

  results_df['Predicted_Label'] = y_pred
  results_df['Correct'] = (results_df['Predicted_Label'] == results_df['True_Label'])

  print( results_df )

def count_missing_values( df, name= 'dataset' ):
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(f'Missing values in the {name}:\n', df.isnull().sum())

def print_classifier_header( classifier='?', stack_depth= 1):
  function_name= '<Unknown Classifier>'
  try:
    function_name= sys._getframe(stack_depth).f_code.co_name
  except e:
    print('Could not lookup function name')

  title= f'({classifier}) {function_name}'

  print('\n' + '='*(8+ len(title)) + '\n' )
  print('    '+ title + '\n')

def classifier_header( classifier ):
  return lambda: print_classifier_header( classifier, stack_depth= 2)
