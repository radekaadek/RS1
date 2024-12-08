import pandas
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

# load data from df.csv
print("Loading data...")
before = time.time()
original_df = pandas.read_csv('df.csv')
print(original_df.info())
after = time.time()
print(f"Time taken: {after - before:.2f} seconds")
# encode building values
le = LabelEncoder()
original_df['building_values'] = le.fit_transform(original_df['building_values'])

# split data into training and testing sets
# df = df.head(500000)
df = original_df.sample(frac=0.02)
print(f"Sum of buildings: {df['building_values'].sum()}")
df_no_buildings = df.drop('building_values', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df_no_buildings, df['building_values'], test_size=0.2, random_state=42)

clf = IsolationForest()
before = time.time()
clf.fit(X_test, y_train)
print(f"Time taken to train model: {time.time() - before:.2f} seconds")
y_pred = clf.predict(X_test)
accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# get a sample of data for testing
df_test = original_df.sample(n=100000)
print(f"Count of testing buildings: {len(df_test)}")
print(f"Sum of testing buildings: {df_test['building_values'].sum()}")
df_test_x = df_test.drop('building_values', axis=1)
df_test_y = df_test['building_values']
predictions = clf.predict(df_test_x)
accuracy = balanced_accuracy_score(df_test_y, predictions)

print(f"Accuracy: {accuracy}")
print(f"Sum of predicted buildings: {predictions.sum()}")
print(f"Sum of actual buildings: {y_test.sum()}")
