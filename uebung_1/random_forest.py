from data_loader import load_csv_from_zip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sklearn.metrics as metrics
import pandas as pd

# Ensure output directory exists
Path("./out").mkdir(parents=True, exist_ok=True)

# Load the data frames from the zip file
data_df, eval_df= load_csv_from_zip('184-702-tu-ml-2025-s-breast-cancer-diagnostic.zip', [
  'breast-cancer-diagnostic.shuf.lrn.csv',
  'breast-cancer-diagnostic.shuf.tes.csv'
])


# Remove non-feature columns and split columns into inputs and output
x = data_df.drop(columns=['class', 'ID'])  
y = data_df['class']

# Create training/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# print( y_test )
 
# Train the model
print('Training...')
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(x_train, y_train)

# Put the test data into the model to see how well it works
y_pred = model.predict(x_test)


# Report some metrics on the model's quality
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))

# Full report
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# Print the true and predicted labels
results_df = x_test.copy()
results_df['True_Label'] = y_test
results_df['Predicted_Label'] = y_pred
results_df['Correct'] = (y_pred != y_test)

print( results_df )

# Let the model predict the labels on the evaluation data set from Kaggle
x_eval = eval_df.drop(columns=['ID'])
y_eval = model.predict(x_eval)

eval_results_df= pd.DataFrame()
eval_results_df['ID']= eval_df['ID']
eval_results_df['class']= y_eval

print( eval_results_df )

# Serialize the results for uploading to Kaggle
eval_results_df.to_csv('./out/breast-cancer-diagnostic.sol.csv', index=False)
