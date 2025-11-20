# Task_04_DA
📞 Telecom Customer Churn Prediction

Machine learning project to predict whether a customer is likely to churn based on behavior, usage, and service attributes.
This repository includes data preprocessing, EDA, model training, evaluation, feature importance, and saved ML models ready for deployment.

🎯 Project Objective

The goal of this project is to predict customer churn for a telecom company using machine learning techniques.
Churn prediction helps businesses identify customers likely to leave and take proactive retention measures.

🧠 Key Steps in the Pipeline
1. Data Loading & Cleaning

Missing values in TotalCharges handled

Removed customerID

Encoded target variable (Churn) using LabelEncoder

2. Feature Engineering

OneHotEncoding for categorical variables

StandardScaler for numerical variables

Imputation for missing data

Preprocessing pipeline saved as preprocessor.joblib

3. Model Building

Two models were trained:

Random Forest Classifier

Logistic Regression

Models saved using joblib:

rf_churn_model.joblib
lr_churn_model.joblib

4. Model Evaluation

Test-set performance:

Model	Accuracy	AUC Score
Random Forest	0.79	0.814
Logistic Regression	0.80	0.836
🔍 Top 10 Important Features (RandomForest)

TotalCharges

tenure

MonthlyCharges

Contract (Month-to-Month)

OnlineSecurity (No)

Electronic Check Payment Method

InternetService (Fiber optic)

TechSupport (No)

SeniorCitizen

OnlineBackup (No)

📘 Jupyter Notebook

The entire workflow (cleaning → preprocessing → training → model saving → inference) is documented in:

👉 churn_notebook.ipynb

🚀 How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Open the Jupyter Notebook
jupyter notebook churn_notebook.ipynb

3. Using the saved model (inference example)
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("rf_churn_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Example: predict for new data
df = pd.read_csv("Telco-Customer-Churn-cleaned.csv").head(5)
X = df.drop("Churn", axis=1)
Xp = preprocessor.transform(X)

predictions = model.predict(Xp)
probabilities = model.predict_proba(Xp)[:, 1]

print(predictions)
print(probabilities)
