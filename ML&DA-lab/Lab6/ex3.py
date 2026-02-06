# Stacking Example for Diabetes datasets
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold


df = pd.read_csv("diabetes.csv")   # Outcome = label
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=3000, random_state=42)
)


models = {
    "SVC": svc,
    "k-NN": knn,
    "Decision Tree": dt,
    "MLP": mlp
}

results = {}

print(" Individual classifiers:\n ")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")


estimators = [
    ('svc', svc),
    ('knn', knn),
    ('dt', dt),
    ('mlp', mlp)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv
)

stacking_scores = cross_val_score(stacking_clf, X, y, cv=cv, scoring='accuracy')
results["Stacking"] = stacking_scores.mean()

print("\nStacking Classifier:\n")
print(f" Mean Accuracy = {stacking_scores.mean():.4f}, Std = {stacking_scores.std():.4f}")


plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Individual Classifiers vs Stacking (Pima Diabetes)")

for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.show()
