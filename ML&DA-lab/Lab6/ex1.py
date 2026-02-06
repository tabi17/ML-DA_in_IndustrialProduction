# exercitiul 1: Bagging cu SVC, k-NN și DT pe Diabetes Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

# Dezactivăm warning-urile pentru a curăța output-ul
warnings.filterwarnings('ignore')

diabetes_file = 'diabetes.csv'
df = pd.read_csv(diabetes_file)

# Separăm features și target
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


classifiers = {
    "SVC (RBF kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree (max_depth=None)": DecisionTreeClassifier(random_state=42)
}


# evaluăm fiecare clasicator individual și cu Bagging
for name, clf in classifiers.items():
    # Bagging cu 10 estimatori
    bagging_clf = BaggingClassifier(estimator=clf, n_estimators=10, random_state=42)

    # Cross-validation pentru clasicator individual
    scores_individual = cross_val_score(clf, X, y, cv=5)
    print(f"{name} → Individual Classifier Accuracy:")
    print(f"Mean: {scores_individual.mean():.4f}, Std: {scores_individual.std():.4f}\n")

    # Cross-validation pentru Bagging
    scores_bagging = cross_val_score(bagging_clf, X, y, cv=5)
    print(f"{name} → Bagging Classifier Accuracy:")
    print(f"Mean: {scores_bagging.mean():.4f}, Std: {scores_bagging.std():.4f}")
    print("=" * 60)

#predicti pentru niste valori aleatoare puse
sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # exemplu aleatoare

print("\n predicti pentru niste valori aleatoare puse\n ")
for name, clf in classifiers.items():
    clf.fit(X, y)
    bagging_clf = BaggingClassifier(estimator=clf, n_estimators=10, random_state=42)
    bagging_clf.fit(X, y)

    pred_individual = clf.predict(sample_input)[0]
    pred_bagging = bagging_clf.predict(sample_input)[0]

    print(f"{name}: Individual → {pred_individual}, Bagging → {pred_bagging}")
