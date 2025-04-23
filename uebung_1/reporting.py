import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import re

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

only_word_chars_pattern = re.compile('[^\\w ]')
def snakeify( text, sep= '_' ):
  return re.sub(only_word_chars_pattern, '', text.lower()).replace(' ', sep)

def export_current_plot_with_title_name( title, xlabel, ylabel ):
  title= snakeify(title)
  xlabel= snakeify(xlabel, '-')
  ylabel= snakeify(ylabel, '-')
  path= f"./out/{title}_{xlabel}_{ylabel}"

  print('Exporting plot:', path)
  plt.savefig(path)

def plot_crossval_scores( scores, title: str, xlabel, ylabel, x_values= None, line_label= None, show= False, stacked= False ):

  # Use numbers from 1...N by default
  if not x_values:
    x_values= range(1,len(scores)+1)

  # Only add legend if we have a label
  line, = plt.plot(x_values, scores)
  if line_label:
    line.set_label( line_label )
    plt.legend(framealpha = 0.6)

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)


  # Export the plot as an image file
  if not stacked:
    export_current_plot_with_title_name( title, xlabel, ylabel)

    if show:
      plt.show()
    else:
      plt.clf()

    
stored_crossval_scores= {}
def store_crossval_scores( classifier, config_name, x_values, scores ):
  storage= stored_crossval_scores.setdefault(classifier, {})

  if config_name in storage:
    raise ValueError(f'Duplicated configuration name {config_name} in classifier {classifier}')
  
  storage[config_name]= (x_values, scores)


def filter_available_score_entries( score_entries, title ):
  available_config_indices= []

  # Find out first which score entries are actually available in the stored data
  for i in range(len(score_entries)):
    (classifier, config, line_label) = score_entries[i]

    if not classifier in stored_crossval_scores:
      print(f'Skipped missing classifier {classifier} when plotting')
      continue

    classifier_data= stored_crossval_scores[classifier]

    if not config in classifier_data:
      print(f'Skipped missing configuration {config} for classifier {classifier} when plotting')
      continue

    available_config_indices.append( i )

  # Nothing to plot
  if len(available_config_indices) < 1:
    print(f'WARNING! Plot is empty: {title}')
  
  return available_config_indices


def plot_stored_crossval_scores( score_entries, score_type, title, xlabel, ylabel, show= False):
  available_config_indices= filter_available_score_entries( score_entries, title )

  # Plot the configuration data as a stacked plot
  for i in range(len(available_config_indices)):
    (classifier, config, line_label) = score_entries[available_config_indices[i]]
    
    config_data= stored_crossval_scores[classifier][config]
    (x_values, score_df)= config_data
    scores= score_df[score_type]

    plot_crossval_scores(
      scores,
      x_values= x_values,
      title= title,
      xlabel= xlabel,
      ylabel= ylabel,
      line_label= line_label,
      stacked= i < (len(available_config_indices) - 1),
      show= show
    )


def plot_stored_crossval_boxplots(score_entries, score_type, title, ylabel, show= False ):
  available_config_indices= filter_available_score_entries( score_entries, title )
  if len(available_config_indices) < 1:
    return

  data= []
  labels= []

  # Gether the configuration data into a 2D list
  for i in range(len(available_config_indices)):
    (classifier, config, line_label) = score_entries[available_config_indices[i]]
    
    config_data= stored_crossval_scores[classifier][config]
    (x_values, score_df)= config_data
    scores= score_df[score_type]

    data.append( scores )
    labels.append( line_label )

  # Plot all boxplots into a single chart
  plt.clf()

  plt.boxplot(data)
  plt.xticks(range(1, len(data)+1),  labels= labels)

  plt.title(title)
  plt.ylabel(ylabel)

  export_current_plot_with_title_name(title, 'boxplots', ylabel)

  if show:
    plt.show()
  else:
    plt.clf()


  



def append_averaged_cv_scores( scores, cv_scores, silent= False ):
  if not silent and len(scores) == 0:
    print('  ', '   -   '.join(list(cv_scores.keys())) )

  for k in cv_scores.keys():
    cv_scores[k]= np.mean(cv_scores[k])

  scores.append(cv_scores)

  if not silent:
    values= [ str(float(x)) for x in list(scores[-1].values()) ]
    print(len(scores), '  '.join(values))
