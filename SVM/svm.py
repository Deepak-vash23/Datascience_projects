# -*- coding: utf-8 -*-
"""SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13jJJ9hr3PsKbFGlPPuKXR2_mYb0-AYf0
"""

# Importing necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


# Load a sample dataset (Iris dataset for binary classification)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features for simplicity
y = (iris.target != 0) * 1  # Binary classification (Iris-setosa vs. others)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the SVM model
model = SVC(kernel='linear')  # Using linear kernel
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# # Visualizing the decision boundary
# plt.figure(figsize=(8, 6))
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', label="Training data")
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='cool', marker='x', label="Predicted test data")

# # Plot the decision boundary
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30),
#                      np.linspace(ylim[0], ylim[1], 30))
# Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot decision boundary and margins
# ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# plt.legend()
# plt.title("SVM Decision Boundary")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()