# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project aims to predict passenger survival on the Titanic using machine learning models. We preprocess the dataset, engineer relevant features, and apply multiple machine learning algorithms to achieve high accuracy in classification.

## 📂 Dataset
The dataset used in this project is the Titanic dataset from Kaggle, which contains details about passengers, such as age, sex, class, and whether they survived or not.

## 🛠️ Technologies Used
- Python
- Pandas & NumPy (Data Preprocessing & Manipulation)
- Scikit-learn (Machine Learning Models)
- Matplotlib & Seaborn (Data Visualization)
- Jupyter Notebook / Python Scripts

## 🚀 Project Workflow

### **Step 1: Data Preprocessing**
- Loaded the dataset and explored missing values.
- Handled missing values by filling them with median or mode.
- Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).

### **Step 2: Feature Engineering**
- Converted categorical features (`Sex`, `Embarked`) into numerical form.
- One-hot encoded the `Embarked` column.
- Ensured all data was in numerical format for model training.

### **Step 3: Feature Selection**
- Selected relevant features such as `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked_Q`, and `Embarked_S`.
- Removed highly correlated and redundant features.

### **Step 4: Model Selection & Training**
- Trained and evaluated multiple models:
  - **Logistic Regression**
  - **Random Forest** ✅ (Chosen model)
  - **Support Vector Machine (SVM)**

### **Step 5: Hyperparameter Tuning**
- Tuned the Random Forest model using Grid Search to improve accuracy.
- Best parameters found: `{ 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200 }`

### **Step 6: Model Evaluation**
- Evaluated model performance using:
  - **Accuracy Score**
  - **Precision, Recall, F1-score**
  - **Confusion Matrix**

### **Step 7: Error Analysis**
- Analyzed misclassified samples.
- Identified patterns in incorrect predictions to improve model performance.

### **Step 8: Feature Importance Analysis**
- Used feature importance scores from the Random Forest model.
- Visualized the most influential features for survival prediction.

## 🎯 Results
- **Final Model: Random Forest (Tuned)**
- **Final Accuracy: 83.24%**
- **Confusion Matrix & Classification Report showed improved precision and recall**

## 📌 How to Run the Project
   Clone this repository:
   ```bash
   git clone https://github.com/VarunSharma189/Titanic-Data-Science-Project.git
   ```

## 📌 Future Enhancements
- Try deep learning models like Neural Networks.
- Explore advanced feature engineering techniques.
- Implement automated hyperparameter tuning with `Optuna` or `Bayesian Optimization`.

## 📌 Author
Developed by **Varun Sharma** 🚀

## 📌 License
This project is open-source and available under the MIT License.
