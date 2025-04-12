import sklearn.metrics as metrics


def report(y_test, y_pred):
  # Report some metrics on the model's quality
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
  print("Precision:", metrics.precision_score(y_test, y_pred))
  print("Recall:", metrics.recall_score(y_test, y_pred))
  print("F1 Score:", metrics.f1_score(y_test, y_pred))

  # Full report
  print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

def compare_labels(x_test, y_test, y_pred):
  results_df = x_test.copy()
  results_df['True_Label'] = y_test

  results_df.reset_index(drop=True, inplace=True)

  results_df['Predicted_Label'] = y_pred
  results_df['Correct'] = (results_df['Predicted_Label'] == results_df['True_Label'])

  print( results_df )
