# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# # Load the preprocessed dataset (Make sure the path is correct)
# train_data = pd.read_csv("processed_titanic.csv")  # If you've saved the preprocessed data

# # Define features (X) and target variable (y)
# X = train_data.drop(columns=["Survived"])  # Features
# y = train_data["Survived"]  # Target variable

# # Split the data into training (80%) and testing (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],   # Number of trees
#     'max_depth': [None, 10, 20, 30],   # Maximum depth of trees
#     'min_samples_split': [2, 5, 10],   # Min samples required to split a node
#     'min_samples_leaf': [1, 2, 4]      # Min samples in leaf node
# }

# # Initialize the model
# rf = RandomForestClassifier(random_state=42)

# # Perform GridSearchCV
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,                 
#     n_jobs=-1,            
#     verbose=2
# )

# # Train using GridSearch
# grid_search.fit(X_train, y_train)

# # Print best parameters
# print("Best Parameters:", grid_search.best_params_)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed dataset
train_data = pd.read_csv("processed_titanic.csv")  

# Define features (X) and target variable (y)
X = train_data.drop(columns=["Survived"])
y = train_data["Survived"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model with Best Parameters
rf_best = RandomForestClassifier(
    max_depth=20, 
    min_samples_leaf=1, 
    min_samples_split=10, 
    n_estimators=200, 
    random_state=42
)

rf_best.fit(X_train, y_train)  # Train the model

# Predictions
y_pred = rf_best.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"🚀 Tuned Random Forest Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🌀 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))