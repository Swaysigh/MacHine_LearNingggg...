import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
df = pd.read_csv(r"C:\\Users\\assen\\Downloads\\iris\\iris.data", header=None)
df.columns = ["sepal_length","sepal_width","petal_length","petal_width","species"]

# Explore the dataset
print(df.head(10))
print(df.isnull().sum())
print(df['species'].value_counts())

# Encode species as numerical values( a.k.a Label Encoding)
df['species'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
print(df.head(10))

# Define feature columns and target variable
feature_columns = ["sepal_length","sepal_width","petal_length","petal_width"]
X = df[feature_columns]
Y = df["species"]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=200)
print("\n Training the model... \n")
model.fit(X_train, Y_train)
print("\n Model training completed. \n")

# Make predictions
y_pred = model.predict(X_test)
print("\n Predictions on test data: \n", y_pred)

# Evaluate the model
accuracy = model.score(X_test, Y_test)
print("\n Model Accuracy: {:.2f}%".format(accuracy * 100))
print("\n Confusion Matrix: \n", confusion_matrix(Y_test, y_pred))
print("\n Classification Report: \n", classification_report(Y_test, y_pred))
