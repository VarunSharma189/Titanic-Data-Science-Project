# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset 
train_data = pd.read_csv("processed_titanic.csv")  

# Select the features (X) and labels (y) from the dataset
X = train_data.drop(columns=["Survived"])  # All features except 'Survived'
y = train_data["Survived"]  # The target variable 'Survived'

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model with the training data
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Model Accuracy: {accuracy_rf:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

import joblib

# Save the trained model
joblib.dump(rf_model, "Random-Forest.pkl")
print("✅ Model saved successfully as 'Random-Forest.pkl'")