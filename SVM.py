# Import necessary libraries for SVM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
train_data = pd.read_csv("processed_titanic.csv")  # Ensure you are loading the processed dataset

# Define features (X) and target variable (y)
X = train_data.drop(columns=["Survived"])  # Independent variables
y = train_data["Survived"]  # Target variable

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier with a linear kernel
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))