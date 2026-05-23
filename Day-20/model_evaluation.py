import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv('creditcard.csv')

# Features and target
X = data.drop('Class', axis=1)

y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# Decision Tree
# ==============================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

# ==============================
# Random Forest
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ==============================
# Metrics Function
# ==============================

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"\n{name} Performance")

    print("-" * 40)

    print("Accuracy :", accuracy)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("F1 Score :", f1)

    print("ROC-AUC  :", roc_auc)

    print("\nClassification Report:\n")

    print(classification_report(y_true, y_pred))

# ==============================
# Evaluate Models
# ==============================

evaluate_model(
    "Decision Tree",
    y_test,
    dt_predictions
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

# ==============================
# Confusion Matrix - Decision Tree
# ==============================

dt_cm = confusion_matrix(
    y_test,
    dt_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt='d'
)

plt.title("Decision Tree Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==============================
# Confusion Matrix - Random Forest
# ==============================

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==============================
# Metrics Comparison Table
# ==============================

results = pd.DataFrame({

    'Model': ['Decision Tree', 'Random Forest'],

    'Accuracy': [
        accuracy_score(y_test, dt_predictions),
        accuracy_score(y_test, rf_predictions)
    ],

    'Precision': [
        precision_score(y_test, dt_predictions),
        precision_score(y_test, rf_predictions)
    ],

    'Recall': [
        recall_score(y_test, dt_predictions),
        recall_score(y_test, rf_predictions)
    ],

    'F1 Score': [
        f1_score(y_test, dt_predictions),
        f1_score(y_test, rf_predictions)
    ],

    'ROC-AUC': [
        roc_auc_score(y_test, dt_predictions),
        roc_auc_score(y_test, rf_predictions)
    ]

})

print("\nMetrics Comparison Table:\n")

print(results)

# ==============================
# Final Conclusion
# ==============================

print("""

Why Accuracy Alone is Misleading:

Fraud datasets are highly imbalanced.

A model may achieve very high accuracy
by predicting most transactions as non-fraud.

Precision, Recall, F1-score and ROC-AUC
provide better evaluation for fraud detection systems.

""")