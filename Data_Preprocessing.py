# import pandas as pd
# import numpy as np

# # Load the dataset (Ensure the correct path)
# train_data = pd.read_csv("train.csv")  

# # Fill missing Age values with the median age
# train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

# # Fill missing Embarked values with the most common port (mode)
# train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

# # Drop unnecessary columns (Cabin, Name, Ticket, PassengerId)
# train_data = train_data.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# # ✅ Print to check the first few rows
# print("First 5 rows after preprocessing:")
# print(train_data.head())

# # ✅ Print to check missing values
# print("\nMissing values after preprocessing:")
# print(train_data.isnull().sum())


import pandas as pd
import numpy as np

# Load the dataset (Ensure the correct path)
train_data = pd.read_csv("train.csv")  

# Fill missing Age values with the median age
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

# Fill missing Embarked values with the most common port (mode)
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

# Drop unnecessary columns (Cabin, Name, Ticket, PassengerId)
train_data = train_data.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Convert 'Sex' column to numerical (male = 0, female = 1)
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})

# Convert 'Embarked' column to numerical using one-hot encoding
train_data = pd.get_dummies(train_data, columns=["Embarked"], drop_first=True)

# ✅ Print to check the first few rows
print("First 5 rows after preprocessing:")
print(train_data.head())

# ✅ Print to check missing values
print("\nMissing values after preprocessing:")
print(train_data.isnull().sum())