import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

df = pd.read_csv(r"C:\\Users\\assen\\Downloads\\iris\\iris.data", header=None)
df.columns = ["sepal_length","petal_length","sepal_width","petal_width","class"]
df["class"] = df["class"].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})

Feature_columns = ["sepal_length","petal_length","sepal_width","petal_width"]
x = df[Feature_columns]
Y = df["class"]

x_train,x_test,y_train,y_test = train_test_split(x,Y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred  = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print("\n Model Accuracy: {:.2f}%".format(accuracy * 100))
print("\n Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\n Classification Report: \n", classification_report(y_test, y_pred))

from sklearn.tree import plot_tree

print("\nDisplaying the Decision Tree...")

# We need the text names for the features and classes for the plot
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Set the figure size
plt.figure(figsize=(15,10))

# Create the plot
plot_tree(model, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True)

# Show the plot
plt.show()

# Create your new flower data
new_flower_data = [[6.0, 2.5, 4.0, 1.1]] # This should predict Iris-versicolor

# Put it in a DataFrame with the correct column names
new_flower_df = pd.DataFrame(new_flower_data, columns=Feature_columns)

# Now predict using the DataFrame
prediction = model.predict(new_flower_df)

species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
predicted_species = species_names[prediction[0]]

print(f"\n--- New Flower Prediction (with no warning) ---")
print(f"The model predicts your new flower is: {predicted_species}")