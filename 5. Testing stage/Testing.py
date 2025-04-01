import pandas as pd
import joblib  # To load the trained model

# Load the trained model
rf_model = joblib.load("Random-Forest.pkl")  # Ensure this file exists

# Load the new test dataset
test_data = pd.read_csv("test_input.csv")

# Predict survival using the trained model
predictions = rf_model.predict(test_data)

# Add predictions to the test file
test_data["Predicted_Survival"] = predictions

# Save the results
test_data.to_csv("test_predictions.csv", index=False)

print("Predictions saved successfully! 🎯")
print(test_data)
