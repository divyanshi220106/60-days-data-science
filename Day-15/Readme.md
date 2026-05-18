# Customer Churn Prediction using Logistic Regression

## Project Overview

This project predicts customer churn using Machine Learning Classification techniques.  
The model identifies customers who are likely to leave a subscription-based service.

Customer churn prediction helps businesses:
- Improve customer retention
- Reduce revenue loss
- Identify high-risk customers early
- Optimize marketing strategies

---

# Objective

Build a Logistic Regression classification model to predict whether a customer will churn or not.

---

# Dataset

Dataset Used:
Telco Customer Churn Dataset

Features include:
- Gender
- Senior Citizen
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Payment Method
- Churn Status

Dataset Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

# Machine Learning Workflow

## 1. Data Loading
Loaded customer churn dataset into Pandas DataFrame.

## 2. Data Cleaning
- Handled missing values
- Converted TotalCharges to numeric datatype

## 3. Feature Encoding
Categorical columns encoded using LabelEncoder.

## 4. Feature & Target Selection
- Features (X)
- Target Variable (y = Churn)

## 5. Train-Test Split
Dataset split into:
- 80% Training Data
- 20% Testing Data

## 6. Model Training
Trained Logistic Regression classifier.

## 7. Predictions
Generated predictions on unseen customer data.

## 8. Model Evaluation
Evaluated model using:
- Accuracy Score
- Classification Report
- Confusion Matrix

---

# Logistic Regression Formula

The model predicts probability using the sigmoid function:

P(Y=1) = 1 / (1 + e^-(β0 + β1x1 + ... + βnxn))

---

# Confusion Matrix

The confusion matrix helps analyze:
- True Positives
- True Negatives
- False Positives
- False Negatives

---

# Business Implications

## False Positive
Customer predicted to leave but actually stays.
- Unnecessary retention offers
- Increased operational cost

## False Negative
Customer predicted to stay but actually leaves.
- Revenue loss
- Customer loss
- Most critical business error

---

# Results

The model successfully:
- Classified churn customers
- Generated prediction results
- Visualized confusion matrix
- Identified important churn factors

---

# Project Structure

```text
Customer-Churn-Prediction/
│
├── README.md
├── churn_prediction.ipynb
├── churn_predictions.csv
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── confusion_matrix.png
```

---

# How to Run

## Step 1
Clone repository:

```bash
git clone <repository_link>
```

## Step 2
Open Google Colab or Jupyter Notebook.

## Step 3
Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Step 4
Run the notebook cells sequentially.

---

# Learning Outcomes

Through this project, I learned:
- Classification fundamentals
- Logistic Regression
- Data preprocessing
- Feature encoding
- Model evaluation
- Confusion matrix interpretation
- Business understanding of ML predictions

---

# Future Improvements

- Hyperparameter tuning
- SMOTE for class imbalance
- Random Forest & XGBoost comparison
- Feature engineering
- Deployment using Flask or Streamlit

---

# Author

Data Science Learning Project — Day 15
