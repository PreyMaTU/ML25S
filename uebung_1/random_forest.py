from reporting import eval_prediction, print_classifier_header
from dataset_heart_disease import encode_dataset_heart_disease, prepare_numeric_dataset_heart_disease
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def export_kaggle_results(ids_eval, y_eval):
  eval_results_df= pd.DataFrame()
  eval_results_df['ID']= ids_eval
  eval_results_df['class']= y_eval

  # print( eval_results_df )

  # Serialize the results for uploading to Kaggle
  eval_results_df.to_csv('./out/breast-cancer-diagnostic.sol.csv', index=False)


############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_version_01( x, y, x_eval, ids_eval ):
  print_classifier_header()

  # Create training/test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


  # Train the model
  print('Training...')
  model = RandomForestClassifier(n_estimators=50, random_state=42)
  model.fit(x_train, y_train)

  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test)

  eval_prediction( x_test, y_test, y_pred)


  # Let the model predict the labels on the evaluation data set from Kaggle
  y_eval = model.predict(x_eval)

  export_kaggle_results(ids_eval, y_eval)



def dataset_breast_cancer_version_02( x, y, x_eval, ids_eval ):
  print_classifier_header()

  pass

############################################################################################
# Dataset Loan:
#TODO:

############################################################################################
# Dataset Dota:
#TODO:

############################################################################################
# Dataset Heart Disease:


def dataset_heart_disease_version_01( x, y ):
  print_classifier_header()
  
  x, y= encode_dataset_heart_disease( x, y )

  # Create training/test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  x_train, x_test = prepare_numeric_dataset_heart_disease(x_train, x_test, scale_values= False)

  # Train the model
  print('Training...')
  model = RandomForestClassifier(n_estimators=50, random_state=42)
  model.fit(x_train, y_train)

  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test)

  eval_prediction( x_test, y_test, y_pred, True)


