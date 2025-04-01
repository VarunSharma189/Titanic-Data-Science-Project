import pandas as pd
import numpy as np
import joblib  # To load the trained model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the preprocessed dataset
train_data = pd.read_csv("processed_titanic.csv")  # Ensure you save preprocessed data earlier

# Split the dataset again (ensure consistency)
X = train_data.drop(columns=["Survived"])  
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
rf_model = joblib.load("Random-Forest.pkl")  # Ensure you save your trained model earlier

# Predict using the model
y_pred = rf_model.predict(X_test)

# Perform error analysis
misclassified = X_test.copy()
misclassified["Actual"] = y_test.values
misclassified["Predicted"] = y_pred
misclassified = misclassified[y_test != y_pred]

print("Misclassified Samples:")
print(misclassified.head())

# Show confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
