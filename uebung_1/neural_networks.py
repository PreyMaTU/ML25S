
from dataset_loan import encode_dataset_loan, prepare_numeric_dataset_loan
from dataset_heart_disease import encode_dataset_heart_disease, prepare_numeric_dataset_heart_disease
from dataset_dota import encode_dataset_dota

from reporting import eval_prediction, classifier_header
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed

import pandas as pd
import numpy as np

header= classifier_header('NN')

############################################################################################
# Dataset Breast Cancer:

def dataset_breast_cancer_version_01( x ,y, x_eval, ids_eval ):
  header()

  set_random_seed(42)

  # Convert types of columns to numerical values
  y = y.astype(int)  # convert True/False to 1/0

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y,  random_state=42)

  # Scale/Normalize
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)

  # Build model
  model = Sequential([
      Dense(12, activation='relu', input_shape=(x_train_scaled.shape[1],)),
      #Dense(64, activation='relu'),
      #Dense(16, activation='relu'),
      #Dense(16, activation='relu'),
      #Dense(1, activation='sigmoid')  # binary classification
      Dense(2, activation='softmax')  # 2 classes

  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Train
  model.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), epochs=20, batch_size=32)

  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test_scaled)
  pred_classes = np.argmax(y_pred, axis=1)
  
  eval_prediction( x_test, y_test, pred_classes )


def dataset_breast_cancer_version_02( x, y, x_eval, ids_eval ):
  header()
  
  pass


############################################################################################
# Dataset Loan:

def dataset_loan_version_01( x, y, x_eval, ids_eval ):
  header()

  set_random_seed(42)

  x, y= encode_dataset_loan( x, y )

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  # Scale + Outlier
  x_train, x_test = prepare_numeric_dataset_loan(x_train, x_test)


  # Build model
  model = Sequential([
      Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(7, activation='softmax')  # 7 classes
  ])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # Train
  model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)
  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test)
  pred_classes = np.argmax(y_pred, axis=1)
  
  #labels = label_encoder.inverse_transform(pred_classes)
  eval_prediction( x_test, y_test, pred_classes, True )
  



############################################################################################
# Dataset Dota:

def dataset_dota_version_01( x, y ):
  header()
  set_random_seed(42)

  x, y= encode_dataset_dota( x, y )

  x = x.astype(float)
  y = y.astype(float)

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  # Build model
  model = Sequential([
      Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(2, activation='softmax')  # 2 classes
  ])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  

  # x_train= np.asarray(x_train).astype(np.float32)
  # y_train= np.asarray(y_train).astype(np.float32)
  # x_test= np.asarray(x_test).astype(np.float32)
  # y_test= np.asarray(y_test).astype(np.float32)
  print( x_train )

  # Train
  model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)
  
  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test)
  pred_classes = np.argmax(y_pred, axis=1)
  #labels = label_encoder.inverse_transform(pred_classes)
  eval_prediction( x_test, y_test, pred_classes, True )
  


############################################################################################
# Dataset Heart Disease:

def dataset_heart_disease_version_01( x, y ):
  header()
  set_random_seed(42)

  x, y= encode_dataset_heart_disease( x, y )

  # Train-test split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

  # Scale + Outlier
  x_train, x_test = prepare_numeric_dataset_heart_disease(x_train, x_test)

  # Build model
  model = Sequential([
      Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(5, activation='softmax')  # 5 classes (0-4)
  ])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  # Train
  model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)
  
  # Put the test data into the model to see how well it works
  y_pred = model.predict(x_test)
  pred_classes = np.argmax(y_pred, axis=1)
  
  #labels = label_encoder.inverse_transform(pred_classes)
  eval_prediction( x_test, y_test, pred_classes, True )
  
