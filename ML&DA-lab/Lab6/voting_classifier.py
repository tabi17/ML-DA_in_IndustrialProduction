import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


svc = SVC(kernel = 'linear')
knn =KNeighborsClassifier(n_neighbors=5)
dt  =DecisionTreeClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

Base_learner = [
    ('svc', svc),
    ('knn', knn),
    ('dt', dt),
    ('mlp', mlp)
]
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])


# --- HARD VOTING ---
eclf1 = VotingClassifier(estimators=Base_learner, voting='hard')
eclf1 = eclf1.fit(X, y)
print("Hard voting predictions:", eclf1.predict(X))

# --- SOFT VOTING ---
# Requires all models to have predict_proba()
eclf2 = VotingClassifier(estimators=Base_learner, voting='soft')
eclf2 = eclf2.fit(X, y)
print("Soft voting predictions:", eclf2.predict(X))