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

📊 Steps Performed
1️⃣ Data Collection
Loaded train.csv and test.csv using pandas.

Checked data types, missing values, and basic statistics.

2️⃣ Exploratory Data Analysis (EDA)
Visualized distributions of Survived, Pclass, Age, Sex, etc.

Checked correlations between features.

Analyzed missing values in Age, Cabin, and Embarked.

3️⃣ Data Preprocessing
Handled missing values:

Age → Filled with median age.

Embarked → Filled with most common port.

Cabin → Dropped due to excessive missing data.

Converted categorical data (Sex, Embarked) using One-Hot Encoding.

Dropped irrelevant columns (Name, Ticket, PassengerId).

4️⃣ Feature Selection
Selected important features:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S.

5️⃣ Model Selection
Tested different algorithms:

✅ Logistic Regression

✅ Random Forest (Final Model)

✅ Support Vector Machine (SVM)

6️⃣ Model Training & Evaluation
Split data into train (80%) and test (20%).

Trained models and evaluated performance using:

Accuracy, Precision, Recall, F1-score, Confusion Matrix.

Final Random Forest Model Accuracy: 81.01%.

7️⃣ Hyperparameter Tuning
Tuned n_estimators, max_depth, min_samples_split, etc., using GridSearchCV.

Optimized Model Accuracy: 83.24% 🚀.

8️⃣ Error Analysis
Analyzed misclassified samples to find common patterns.

9️⃣ Feature Importance
Plotted feature importance from the trained Random Forest model.

📌 Results & Insights
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	81.01%	0.81	0.80	0.80
Random Forest (Tuned)	83.24%	0.83	0.82	0.82
SVM	78.21%	0.78	0.77	0.77
The Random Forest model performed best after tuning.

Gender, Fare, Pclass, and Age were the most important survival predictors.

## 📌 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/VarunSharma189/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python scripts:
   ```bash
   jupyter notebook
   ```
   OR
   ```bash
   python main.py
   ```

## 📌 Future Enhancements
- Try deep learning models like Neural Networks.
- Explore advanced feature engineering techniques.
- Implement automated hyperparameter tuning with `Optuna` or `Bayesian Optimization`.

## 📌 Author
Developed by **Varun Sharma** 🚀

## 📌 License
This project is open-source and available under the MIT License.
