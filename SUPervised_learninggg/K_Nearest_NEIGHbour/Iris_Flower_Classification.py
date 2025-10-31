import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

df = pd.read_csv(r"C:\\Users\\assen\\Downloads\\iris\\iris.data", header=None)
df.columns = ["sepal_length","petal_length","sepal_width","petal_width","class"]
print(df.head(10))

df["class"] = df["class"].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})

Feature_columns = ["sepal_length","petal_length","sepal_width","petal_width"]
x = df[Feature_columns]
Y = df["class"]

x_train,x_test,y_train,y_test = train_test_split(x,Y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred  = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print("\n Model Accuracy: {:.2f}%".format(accuracy * 100))
print("\n Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\n Classification Report: \n", classification_report(y_test, y_pred))
