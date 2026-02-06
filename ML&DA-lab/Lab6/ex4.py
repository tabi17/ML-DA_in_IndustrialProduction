# Hard adn Soft voting Example for Diabetes datasets
import pandas as pd
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

svc = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', probability=True, random_state=42)
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

dt = DecisionTreeClassifier(random_state=42)

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(10,),
                  max_iter=1000,
                  random_state=42)
)

estimators = [
    ('svc', svc),
    ('knn', knn),
    ('dt', dt),
    ('mlp', mlp)
]


print(" Individual classifiers:\n")
for name, clf in estimators:
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"{name}: Mean accuracy = {scores.mean():.4f}")

#
hard_voting = VotingClassifier(
    estimators=estimators,
    voting='hard'
)

scores_hard = cross_val_score(hard_voting, X, y, cv=5, scoring='accuracy')
print("\nHard Voting:\n")
print(f"Mean accuracy = {scores_hard.mean():.4f}")

#
soft_voting = VotingClassifier(
    estimators=estimators,
    voting='soft'
)

scores_soft = cross_val_score(soft_voting, X, y, cv=5, scoring='accuracy')
print("\nSoft Voting:\n")
print(f"Mean accuracy = {scores_soft.mean():.4f}")
