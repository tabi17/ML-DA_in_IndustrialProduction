# AdaBoost Example for Iris and Pima Indians Diabetes datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_iris

# --- Load datasets ---
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

diabetes_file = 'diabetes.csv'  # modifică calea dacă ai altă locație
diabetes_df = pd.read_csv(diabetes_file)
X_diabetes = diabetes_df.drop('Outcome', axis=1).values
y_diabetes = diabetes_df['Outcome'].values

# --- Define a weak learner for AdaBoost ---
dt_base = DecisionTreeClassifier(max_depth=1, random_state=42)  # weak learner

# --- Function to evaluate AdaBoost ---
def evaluate_adaboost(X, y, base_estimator, dataset_name):
    clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"\nAdaBoost with {base_estimator.__class__.__name__} on {dataset_name}:")
    print(f"  Mean Accuracy: {scores.mean():.4f}")
    print(f"  Std Accuracy : {scores.std():.4f}")
    return scores.mean()

# --- Evaluate on both datasets ---
results = {}
results["Iris"] = evaluate_adaboost(X_iris, y_iris, dt_base, "Iris")
results["Diabetes"] = evaluate_adaboost(X_diabetes, y_diabetes, dt_base, "Pima Diabetes")

# --- Plot results ---
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), results.values(), color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("AdaBoost Accuracy with DecisionTree Base Learner")
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()
