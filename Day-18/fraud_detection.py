import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Load dataset
data = pd.read_csv('creditcard.csv')

# Display dataset
print(data.head())

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

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

# Decision Tree Accuracy
print("\nDecision Tree Accuracy:")

print(
    accuracy_score(y_test, dt_predictions)
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:")

print(
    accuracy_score(y_test, rf_predictions)
)
print("\nClassification Report:\n")

print(
    classification_report(y_test, rf_predictions)
)
importance = rf_model.feature_importances_

feature_names = X.columns

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:\n")

print(feature_importance.head(10))
top_features = feature_importance.head(10)

plt.figure(figsize=(10,5))

plt.bar(
    top_features['Feature'],
    top_features['Importance']
)

plt.xticks(rotation=90)

plt.title("Top 10 Important Features")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()
