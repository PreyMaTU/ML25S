import json
import pandas as pd

stored_crossval_scores= {}
def store_crossval_scores( classifier, config_name, x_values, scores ):
  storage= stored_crossval_scores.setdefault(classifier, {})

  if config_name in storage:
    raise ValueError(f'Duplicated configuration name {config_name} in classifier {classifier}')
  
  storage[config_name]= (x_values, scores)

def export_stored_crossval_scores( name ):
  path= f'data/{name}.json'
  print('Exporting scores file:', path)

  json_object= {}
  for classifier in stored_crossval_scores:
    classifier_data= stored_crossval_scores[classifier]

    for config in classifier_data:
      config_data= classifier_data[config]

      x_values, score_df= config_data
      score_dict= score_df.reset_index().to_dict(orient='list')

      json_object.setdefault(classifier, {})[config]= {
        'x_values': x_values,
        'scores': score_dict
      }

  with open(path, 'w') as outfile:
    json.dump(json_object, outfile)

def import_stored_crossval_scores( name ):
  path= f'data/{name}.json'
  print('Importing scores file:', path)

  json_object= None
  with open(path, 'r') as infile:
    json_object= json.load(infile)

  for classifier in json_object:
    json_classifier_data= json_object[classifier]
    classifier_data= stored_crossval_scores.setdefault(classifier, {})

    for config in json_classifier_data:
      if config in classifier_data:
        print(f'WARNING! Imported scores file {name} overrides config "{config}" in classifier {classifier}')

      # Get the x_values and convert the scores into a dataframe
      json_config_data= json_classifier_data[config]
      scores_df= pd.DataFrame( json_config_data['scores'] )

      # Store the config data as a tuple
      classifier_data[config]= (json_config_data['x_values'], scores_df)
