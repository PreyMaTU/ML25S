from data_loader import load_csv_from_zip
from reporting import report, compare_labels
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import set_random_seed


# For reproducibility
set_random_seed(42)

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

# TODO: Handle missing values
# data_df = data_df.dropna()  # or use fillna()

# Convert types of columns to numerical values
y = y.astype(int)  # convert True/False to 1/0

# Train-test split (only needed if you don’t already have test set)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), epochs=20, batch_size=32)

# Put the test data into the model to see how well it works
y_pred = model.predict(x_test_scaled)

y_pred= y_pred.round().astype(int)

# Report some metrics on the model's quality
report(y_test, y_pred)

compare_labels(x_test, y_test, y_pred)

