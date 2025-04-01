import pandas as pd

# Load the dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Display first few rows
print(train_data.head())

# Check data information
print(train_data.info())

# Check for missing values
print(train_data.isnull().sum())  