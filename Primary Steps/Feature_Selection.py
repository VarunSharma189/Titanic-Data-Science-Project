import pandas as pd
import numpy as np

# Load the dataset
train_data = pd.read_csv("train.csv")

# Fill missing values
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

# Drop unnecessary columns
train_data = train_data.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Convert 'Sex' to numerical
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})

# One-Hot Encoding for 'Embarked' column
train_data = pd.get_dummies(train_data, columns=["Embarked"], drop_first=True)

# Convert True/False to 0/1
train_data["Embarked_Q"] = train_data["Embarked_Q"].astype(int)
train_data["Embarked_S"] = train_data["Embarked_S"].astype(int)

# Select features and target variable
target = 'Survived'
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']

# Prepare the feature matrix (X) and target vector (y)
X = train_data[features]
y = train_data[target]

# ✅ Print to confirm
print("Features Selected for Model Training:")
print(X.head())

print("\nTarget (Survived):")
print(y.head())

train_data.to_csv("processed_titanic.csv", index=False)
