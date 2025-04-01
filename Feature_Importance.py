import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the preprocessed data (ensure the correct path)
train_data = pd.read_csv("processed_titanic.csv")

# Define your features (X) and target (y) columns
X = train_data.drop(columns=["Survived"])  # All features except 'Survived'
y = train_data["Survived"]  # Target variable

# Load the best Random Forest model
rf_best = joblib.load("Random-Forest.pkl")

# Feature Importances
feature_importances = rf_best.feature_importances_

# Create a DataFrame with feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance in Random Forest Model")
plt.show()